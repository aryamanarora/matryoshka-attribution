from typing import Callable, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from einops import einsum

from .graph import Graph
from .utils import tokenize_plus, compute_mean_activations
from .evaluate import evaluate_baseline, evaluate_graph


def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor], neuron:bool=False,
                            per_example:bool=False):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)
        per_example (bool): keep the batch axis in the score update -- `scores` is then
            [batch, n_forward] ([batch, n_forward, d_model] with neuron=True) and holds one
            estimate per example, for methods that take |.| per example before averaging (AtP).

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, 
        i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), 
        while the second set will subtract out the activations they are run on (run these on clean input). 
        The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
    """
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
        
    # Fills up the activation difference matrix. In the default case (not separate_activations), 
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations:torch.Tensor, hook: HookPoint, add:bool=True):
        acts = activations.detach()
        try:
            if add:
                activation_difference[:, :, index] += acts
            else:
                activation_difference[:, :, index] -= acts

        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook: HookPoint):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        if gradients is None:
            # No gradient reached this node: under GradDrop the nodes inside the dropped block
            # are cut off from the loss and autograd hands their hooks an undefined (None) grad.
            # That IS the zero term of the paper's sum, so contribute nothing.
            return
        grads = gradients.detach()
        try:
            if per_example:
                pat = ('batch pos ... hidden, batch pos ... hidden -> batch ... hidden' if neuron
                       else 'batch pos ... hidden, batch pos ... hidden -> batch ...')
                scores[:, fwd_index] += einsum(activation_difference[:, :, fwd_index], grads, pat)
            elif neuron:
                s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ... hidden')
                scores[fwd_index] += s
            else:
                s = einsum(activation_difference[:, :, fwd_index], grads,'batch pos ... hidden, batch pos ... hidden -> ...')
                scores[fwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(fwd_index, bwd_index, scores.size())
            raise e

    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))
    
    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                     intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                     intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    graph.nodes_in_graph[:] = True
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    nodes = graph.nodes.values() if quiet else tqdm(graph.nodes.values())
    for node in nodes:
        for edge in node.child_edges:
            edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention, 
                                                intervention_dataloader=intervention_dataloader, quiet=True, skip_clean=True).mean().item()
        node.score = intervened_performance - baseline
        for edge in node.child_edges:
            edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.nodes_scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                      steps=30, quiet:bool=False, neuron:bool=False,
                      intervention: Literal['patching', 'zero']='patching'):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            # ZERO ablation: skip the corrupted forward. activation_difference starts at 0, so
            # the "corrupted" activations (and the input path's start) are 0 and the clean hooks
            # below leave 0 - clean -- the endpoint delta of the zero intervention the circuit
            # is then scored under (evaluate.py intervention='zero').
            if intervention == 'patching':
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        # + activations * 0  will cause a backwards pass on new_input
        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_eap_ig_mc(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                         steps=1, quiet:bool=False, neuron:bool=False, seed:int=0,
                         intervention: Literal['patching', 'zero']='patching'):
    """"Stepless" EAP-IG-inputs: alpha ~ U(0,1) drawn per example instead of a fixed grid.

    EAP-IG-inputs estimates  (a_clean - a_corrupted) . integral_0^1 grad(alpha) d alpha  with a
    RIGHT-ENDPOINT Riemann sum over alpha = k/m, k = 1..m, at m backward passes per batch. This
    estimates the SAME integral by Monte Carlo -- draw alpha ~ U(0,1), average -- which is
    unbiased at every m, including m = 1.

    THAT IS THE POINT OF m=1. get_scores_eap_ig(steps=1) evaluates the grid at its single point
    alpha = 1, i.e. the gradient at the CLEAN input; it is not an estimate of the integral at all,
    it is input x grad (run_variants.sh calls that row `ig1` and says so). This function at
    steps=1 costs exactly the same one forward+backward per batch and IS an unbiased estimate of
    the m -> infinity limit. So `EAP-IG-inputs-mc --ig-steps 1` vs `EAP-IG-inputs --ig-steps 1` is
    a compute-matched contrast in which the ONLY difference is where alpha is placed.

    ALPHA IS PER-EXAMPLE, WHICH IS FREE AND IS MOST OF THE ESTIMATOR'S QUALITY. The interpolation
    hook writes a [batch, pos, d_model] tensor, so a [batch, 1, 1] alpha costs the same forward as
    a scalar one but yields batch_size independent draws per pass. Since the final score sums over
    examples before anything else, the MC error of a unit's score then falls like
    1/sqrt(n_examples), not 1/sqrt(n_batches) -- on gpt2/ioi that is 1000 draws rather than 50.
    It also means the estimator's variance depends on --batch-size only through arithmetic, not
    through statistics, so a cell forced to batch-size 1 (llama3) is not penalised.

    THE ESTIMAND IS THE INTEGRAL, NOT THE m-STEP GRID, so this does not converge to
    get_scores_eap_ig(steps=m) for the same m -- it converges to steps -> infinity. Comparing it
    against `ref` (m=5) is a comparison of two approximations to the same quantity at 5x different
    cost, which is the interesting comparison; comparing it against m=1 is compute-matched.

    Seeded (default 0) so a run is reproducible and so replicates can be obtained by varying the
    seed alone -- the spread across seeds IS this estimator's error bar, and it is the number that
    decides whether a win over `ig1` is real.
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)

    # CPU generator, so the alpha stream depends only on the seed and the batch sizes -- not on
    # the GPU, the dtype, or anything else that differs between the four models in this sweep.
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            # ZERO ablation: skip the corrupted forward. activation_difference starts at 0, so
            # the "corrupted" activations (and the input path's start) are 0 and the clean hooks
            # below leave 0 - clean -- the endpoint delta of the zero intervention the circuit
            # is then scored under (evaluate.py intervention='zero').
            if intervention == 'patching':
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        # + activations * 0  will cause a backwards pass on new_input
        def input_interpolation_hook(alpha: Tensor):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + alpha * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(steps):
            total_steps += 1
            alpha = torch.rand(batch_size, 1, 1, generator=gen).to(input_activations_corrupted)
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(alpha))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    # Same normalisation as get_scores_eap_ig (total_steps is reset per batch, so it ends at
    # `steps`), which keeps the mc scores on the same scale as the grid ones. Only the ranking is
    # used downstream, but a shared scale makes score-level scatter plots meaningful.
    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_eap_ig_local(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                            steps=30, quiet:bool=False, neuron:bool=False):
    """Node-level EAP-IG (inputs) using the LOCAL activation increment inside the IG sum.

    Standard EAP-IG-inputs scores a node by (a^corrupted - a^clean) . (1/m) sum_k grad(alpha_k),
    pulling the endpoint activation difference out of the integral. That is exact only if the
    node's activation is linear in alpha; but only the INPUT embeddings are interpolated linearly,
    so intermediate activations follow a nonlinear path. This variant instead accumulates the
    realized per-step increment  sum_k grad(alpha_k) . (a(alpha_{k-1}) - a(alpha_k)), i.e. the
    proper Riemann sum along the actual activation trajectory. Gradients are sampled on the same
    grid as node EAP-IG-inputs (alpha = k/steps, k=1..steps), so with steps=1 it is identical.
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # activation_difference is what the bwd hooks dot against grads; we overwrite it each
        # step (via the capture hooks below) with the LOCAL increment (prev - cur) per node.
        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_add, _, _), acts_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, None, neuron=neuron)
        (_, fwd_hooks_sub, _), neg_acts_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, None, neuron=neuron)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_add):
                _ = model(corrupted_tokens, attention_mask=attention_mask)        # acts_corrupted = +corrupted
            with model.hooks(fwd_hooks=fwd_hooks_sub):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)  # neg_acts_clean = -clean
        acts_clean = -neg_acts_clean

        input_idx = graph.forward_index(graph.nodes['input'])
        input_corrupted = acts_corrupted[:, :, input_idx].clone()
        input_clean = acts_clean[:, :, input_idx].clone()

        prev_acts = acts_corrupted.clone()   # a(alpha_0) = corrupted
        cur_acts = acts_corrupted.clone()

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                alpha = k / steps
                new_input = input_corrupted + alpha * (input_clean - input_corrupted) + activations * 0
                cur_acts[:, :, input_idx] = new_input.detach()
                activation_difference[:, :, input_idx] = prev_acts[:, :, input_idx] - new_input.detach()
                return new_input
            return hook_fn

        def capture_hook(index, activations, hook):
            acts = activations.detach()
            cur_acts[:, :, index] = acts
            activation_difference[:, :, index] = prev_acts[:, :, index] - acts

        capture_fwd_hooks = []
        for layer in range(graph.cfg['n_layers']):
            for node in (graph.nodes[f'a{layer}.h0'], graph.nodes[f'm{layer}']):
                capture_fwd_hooks.append((node.out_hook, partial(capture_hook, graph.forward_index(node))))

        for step in range(1, steps + 1):
            fwd_hooks = [(graph.nodes['input'].out_hook, input_interpolation_hook(step))] + capture_fwd_hooks
            with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()
            prev_acts = cur_acts.clone()

    # No /steps: the per-step increment already carries the 1/steps factor.
    scores /= total_items
    return scores


class _DropBlockWrite(torch.autograd.Function):
    """GradDrop's per-layer intervention on the backward pass (AtP*, Kramar et al. 2024, Sec. 3.2):
    identity in value on resid_post, but the incoming gradient is routed ENTIRELY to the block's
    resid_pre (the skip) and NONE of it to the block's write resid_post - resid_pre. That is
    d L / d n under do(block_l out <- its clean value): downstream gradient reaches the layers
    above l only through the residual skip, and every node inside block l gets no gradient at
    all. Forward returns a clone rather than resid_pre + (post - pre).detach(), which would
    perturb the bf16 forward by a rounding error that differs per dropped layer."""
    @staticmethod
    def forward(ctx, resid_post, resid_pre):
        return resid_post.clone()

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def grad_drop_hooks(layer: int):
    """Forward hooks that drop layer `layer`'s residual write from the backward pass (see
    _DropBlockWrite). hook_resid_pre's output is the tensor the block both reads and skips
    forward, so passing it to the Function at hook_resid_post is exactly the skip path."""
    holder = {}
    def keep_pre(t, hook):
        holder['pre'] = t
        return t
    def drop_write(t, hook):
        return _DropBlockWrite.apply(t, holder['pre'])
    return [(f'blocks.{layer}.hook_resid_pre', keep_pre), (f'blocks.{layer}.hook_resid_post', drop_write)]


def get_scores_atp(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                   quiet: bool = False, neuron: bool = False, grad_drop: bool = False,
                   intervention: Literal['patching', 'zero'] = 'patching'):
    """Attribution Patching, AtP / AtP* (Kramar et al. 2024, arXiv:2403.00745), at MIB node granularity.

    AtP (Eq. 4-5 there): for node n and prompt pair (clean, noise), the linear estimate
        I_AtP(n; x) = (n(x_noise) - n(x_clean)) . dL(M(x_clean)) / dn
    and c_AtP(n) = E_x |I_AtP(n; x)|, i.e. the ABSOLUTE VALUE IS TAKEN PER EXAMPLE, inside the
    expectation, so that effects of opposite sign across the distribution do not cancel. The
    signed E_x[I_AtP] is what `EAP` / `EAP-IG-inputs --ig-steps 1` (the I x G row) already
    compute; the per-example |.| is the only thing separating `AtP` from them. A MIB node is
    patched at every position at once, so the per-example estimate is the sum over positions of
    the per-position dot products, and the |.| is taken of that sum -- the linearisation of
    exactly the intervention MIB scores the ranking under.

    AtP* = AtP + the two corrections of Sec. 3:
      * QK fix (Sec. 3.1): recompute the attention softmax with the patched query/key instead of
        linearising through it. It is defined for QUERY and KEY nodes, which are the nodes
        whose first downstream nonlinearity is a saturated softmax. MIB's node set is head
        OUTPUTS (hook_result), MLP outputs and the input embedding -- none of them is a q or k
        node, and each writes linearly into the residual stream -- so on this node set the fix
        has nothing to act on and AtP* reduces to AtP + GradDrop. Stated here so nobody goes
        looking for the missing half.
      * GradDrop (Sec. 3.2), grad_drop=True: for each layer l compute the AtP estimate with the
        gradient through layer l's residual write dropped (dL^l/dn, _DropBlockWrite), then
            c_AtP+GD(n) = E_x [ 1/(L-1) * sum_l |I_AtP+GD_l(n; x)| ]     (their Eq. 12).
        The point is cancellation between a node's direct effect and its indirect effects
        through later layers: with layer l dropped, whichever of those paths goes through l is
        removed, so a node whose effects cancel in the plain gradient shows up in the terms
        where the cancelling path is gone. A node's own layer l(n) contributes a zero term (no
        gradient reaches the inside of a dropped block), so its direct effect is counted in
        L-1 of the L terms, and 1/(L-1) is what makes a node with only a direct effect score
        exactly its AtP value. The input node is in no block and is counted in all L terms, so
        it is divided by L -- the same "number of terms the node survives" rule, extended to the
        one node the paper's node set does not have.
    Cost: 1 (AtP) or L (AtP*) forward+backward passes per batch over the clean prompt, on top of
    the two forwards that cache the clean and noise activations. The paper reuses cached clean
    activations for its L backwards; here each term is a fresh forward with the drop hooks
    installed (the graph differs per l), which is 2x the paper's backward count and still O(L).

    `metric` arrives from run_attribution.py as the batch MEAN, so each example's gradient carries
    a 1/batch factor; it is multiplied back out before the |.| so the score is the per-example
    quantity of the paper and the final /total_items is a plain mean over examples, as in every
    other method here. Accumulation is fp32 (the sums of |.| run to 1000 examples x L terms).
    """
    n_layers = graph.cfg['n_layers']
    shape = (graph.n_forward, graph.cfg['d_model']) if neuron else (graph.n_forward,)
    scores = torch.zeros(shape, device='cuda', dtype=torch.float32)
    input_idx = graph.forward_index(graph.nodes['input'])
    # terms in which each node is NOT inside the dropped block: L-1 for layer nodes, L for input
    n_terms = torch.full((graph.n_forward,), float(n_layers - 1) if grad_drop else 1.0, device='cuda')
    if grad_drop:
        n_terms[input_idx] = float(n_layers)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        per_example = torch.zeros((batch_size,) + shape, device='cuda', dtype=torch.float32)
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, per_example, neuron=neuron, per_example=True)

        with torch.inference_mode():
            if intervention == 'patching':   # ZERO: no noise forward, so the delta is 0 - clean
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)   # += noise acts
            # The clean hooks run ONCE here (-= clean acts -> noise - clean); the gradient passes
            # below must not carry them, or each pass would subtract the clean activations again.
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        batch_acc = torch.zeros_like(per_example)
        for drop in (range(n_layers) if grad_drop else [None]):
            per_example.zero_()
            extra = [] if drop is None else grad_drop_hooks(drop)
            with model.hooks(fwd_hooks=extra, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                (metric_value * batch_size).backward()      # undo the batch mean -> per-example grads
            batch_acc += per_example.abs()
        scores += batch_acc.sum(0)

    scores /= total_items
    scores /= n_terms.view(-1, *([1] * (scores.dim() - 1)))
    return scores.to(model.cfg.dtype)


class ShapleyElementwiseMult(torch.autograd.Function):
    """Elementwise multiply with the RelP/Shapley half-rule: each branch gets 0.5 of the
    incoming gradient (so the product's relevance is split evenly instead of double-counted)."""
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return 0.5 * grad_output * y, 0.5 * grad_output * x


class HalfGrad(torch.autograd.Function):
    """Identity forward, 0.5x gradient backward. Applied to a bilinear matmul's OUTPUT this
    is equivalent to the uniform (half) rule on its inputs: for z = x @ y, scaling grad_z by 0.5
    yields grad_x, grad_y each halved. This matches AttnLRP's uniform rule for bilinear matmuls
    (Achtibat et al. 2024, Eq. 14-15) and LXT's divide_gradient(q,4)/(k,4)/(v,2). Used for the
    RelPShapley QK and OV matmuls."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return 0.5 * grad_output


class ShapleySoftmax(torch.autograd.Function):
    """Shapley-style softmax backward. OURS -- this rule is not from AttnLRP or RelP: LXT
    patches nothing at the softmax, and RelP's rule set is LN/Identity/Half/AH only.
    Forward = softmax; backward
    redistributes relevance proportional to the softmax output and divides by the pre-softmax
    scores: grad_x = (sum_j grad_j * p_j) * p_i / scores_i."""
    @staticmethod
    def forward(ctx, x):
        result = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
        ctx.save_for_backward(x, result)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits, result = ctx.saved_tensors
        with torch.no_grad():
            # upcast to fp32 and guard the divide-by-logits: it blows up to NaN/inf on
            # bf16 / large models (near-zero or extreme attn scores, gemma2 softcapping).
            logits32 = logits.to(torch.float32)
            result32 = result.to(torch.float32)
            R_l = grad_output.to(torch.float32) * result32
            total_R = R_l.sum(-1, keepdim=True)
            safe = logits32.abs() > 1e-6
            grad_x = torch.where(safe, (total_R * result32) / torch.where(safe, logits32, torch.ones_like(logits32)),
                                 torch.zeros_like(logits32))
            grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_x.to(logits.dtype)


class ScaleGrad(torch.autograd.Function):
    """Identity forward, gradient scaled by a constant on backward (GIM Q/K/V grad scaling)."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class TempSoftmax(torch.autograd.Function):
    """GIM softmax: forward = softmax(x) (T=1), backward = softmax(x/T) Jacobian-vector product
    (a 'softer' gradient). Default T=2."""
    @staticmethod
    def forward(ctx, x, T):
        ctx.T = float(T)
        ctx.save_for_backward(x)
        return torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        with torch.no_grad():
            sT = torch.nn.functional.softmax(x / ctx.T, dim=-1, dtype=torch.float32)
            g = grad_output.to(torch.float32)
            dot = (g * sT).sum(-1, keepdim=True)
            grad_x = sT * (g - dot)
        return grad_x.to(x.dtype), None


def _relp_act_coeff(act_fn, pre):
    """Secant linearization of a (gated) activation: coeff = act(pre)/pre, detached.
    For SiLU this equals sigmoid(pre) exactly (silu(z)=z*sigmoid(z)), matching RelP's
    Identity-rule (ModifiedAct transform='identity') and AttnLRP's identity rule for
    element-wise nonlinearities; gate_act = pre*coeff preserves the forward value while
    letting the gradient flow through `pre` with the nonlinearity treated as constant."""
    a = act_fn(pre)
    safe = torch.where(pre.abs() < 1e-6, torch.ones_like(pre), pre)
    return (a / safe).detach()


def build_relp_fwd_hooks(model: HookedTransformer, use_norm=True, use_mlp=True, use_qk=True, shapley_attn=False,
                         softmax_rule=True, linearize_act=True, gim_attn=False, gim_T=2.0,
                         gim_qk_scale=0.25, gim_v_scale=0.5):
    """Forward hooks implementing RelP's three backward modifications (forward values
    unchanged; only the gradient is altered):
      (1) every norm scale is detached -> normalization treated as a constant scaling;
      (2) the gated-MLP gate activation is secant-linearized and the gate x up product
          uses the half-rule (non-gated MLPs just get the linearized activation);
      (3) the attention pattern is detached -> gradient flows only through OV (V path).
    """
    import os
    # explicit args set the default; env vars override (for ablation sweeps)
    use_norm = os.environ.get('RELP_NORM', '1' if use_norm else '0') == '1'
    use_mlp = os.environ.get('RELP_MLP', '1' if use_mlp else '0') == '1'
    use_qk = os.environ.get('RELP_QK', '1' if use_qk else '0') == '1'
    hooks = []
    for name in list(model.hook_dict.keys()):
        if (use_norm and name.endswith('.hook_scale')) or (use_qk and name.endswith('.hook_pattern')):
            hooks.append((name, lambda t, hook: t.detach()))

    # Attention half-rules: HalfGrad on the QK and OV matmul OUTPUTS. Composed, this gives the
    # Q and K gradients a 1/4 factor (1/2 at hook_z -> pattern, then 1/2 again at hook_attn_scores)
    # and the V gradient 1/2, i.e. exactly AttnLRP's uniform rule for bilinear matmuls
    # (Achtibat et al. 2024, Eq. 14-15) as implemented in LXT's divide_gradient(q,4)/(k,4)/(v,2).
    #
    # softmax_rule=True additionally swaps the softmax backward for ShapleySoftmax. That rule is
    # OURS -- LXT patches nothing at the softmax and AttnLRP Eq. (13) reduces to the ordinary
    # softmax Jacobian-vector product -- so faithful AttnLRP is softmax_rule=False.
    # hook_attn_scores fires pre-softmax, so we capture the (scaled+masked) scores there and
    # rebuild the pattern from them when the softmax rule is on.
    if shapley_attn:
        def make_attn_hooks(holder):
            def cap_scores(t, hook):
                s = HalfGrad.apply(t)        # QK matmul half-rule (grad x0.5 to Q and K)
                holder['scores'] = s
                return s
            def shapley_pattern(t, hook):
                return ShapleySoftmax.apply(holder['scores'])   # LRP softmax backward
            def half_z(t, hook):
                return HalfGrad.apply(t)     # OV matmul half-rule (grad x0.5 to pattern and V)
            return cap_scores, shapley_pattern, half_z
        for l in range(model.cfg.n_layers):
            cap, pat, hz = make_attn_hooks({})
            hooks.append((f'blocks.{l}.attn.hook_attn_scores', cap))
            if softmax_rule:
                hooks.append((f'blocks.{l}.attn.hook_pattern', pat))
            hooks.append((f'blocks.{l}.attn.hook_z', hz))

    # GIM attention rules: scale Q/K/V gradients by constants (straight-through) and use a
    # temperature-softmax backward. No MLP rule (GIM leaves the MLP gradient unmodified).
    if gim_attn:
        def make_gim(holder):
            def cap(t, hook):
                holder['scores'] = t
                return t
            def pat(t, hook):
                return TempSoftmax.apply(holder['scores'], gim_T)
            return cap, pat
        for l in range(model.cfg.n_layers):
            cap, pat = make_gim({})
            hooks.append((f'blocks.{l}.attn.hook_q', lambda t, hook: ScaleGrad.apply(t, gim_qk_scale)))
            hooks.append((f'blocks.{l}.attn.hook_k', lambda t, hook: ScaleGrad.apply(t, gim_qk_scale)))
            hooks.append((f'blocks.{l}.attn.hook_v', lambda t, hook: ScaleGrad.apply(t, gim_v_scale)))
            hooks.append((f'blocks.{l}.attn.hook_attn_scores', cap))
            hooks.append((f'blocks.{l}.attn.hook_pattern', pat))

    if not use_mlp:
        return hooks

    def make_store(holder, key):
        def fn(t, hook):
            holder[key] = t
            return t
        return fn

    def make_relp_post(holder, act_fn, b_in, gated):
        def fn(post, hook):
            pre = holder['pre']
            # linearize_act=False keeps the true activation derivative (GIM's
            # activation_backward='grad'); only the gate product's half-rule is applied.
            gate_act = pre * _relp_act_coeff(act_fn, pre) if linearize_act else act_fn(pre)
            if gated:
                out = ShapleyElementwiseMult.apply(gate_act, holder['pre_linear'])
                if b_in is not None:
                    out = out + b_in
                return out
            return gate_act
        return fn

    for l in range(model.cfg.n_layers):
        mlp = model.blocks[l].mlp
        gated = hasattr(mlp, 'W_gate')
        if not linearize_act and not gated:
            continue      # GIM's scale_mlp_gate fires only for gated MLPs; leave gpt2 untouched
        b_in = getattr(mlp, 'b_in', None)
        holder = {}
        hooks.append((f'blocks.{l}.mlp.hook_pre', make_store(holder, 'pre')))
        if gated:
            hooks.append((f'blocks.{l}.mlp.hook_pre_linear', make_store(holder, 'pre_linear')))
        hooks.append((f'blocks.{l}.mlp.hook_post', make_relp_post(holder, mlp.act_fn, b_in, gated)))
    return hooks


def get_scores_relp(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                    quiet: bool = False, neuron: bool = False, relp_hooks: bool = True, detach_qk: bool = True,
                    shapley_attn: bool = False, softmax_rule: bool = True, use_mlp: bool = True,
                    linearize_act: bool = True, gim_attn: bool = False,
                    intervention: Literal['patching', 'zero']='patching'):
    """RelP node attribution: (a^corrupted - a^clean) . grad, a single-point (input x grad)
    attribution where the backward pass uses RelP's relevance rules (see build_relp_fwd_hooks).
    With relp_hooks=False this is exactly input x grad (EAP / 1-step IG) -- used as a self-test."""
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            if intervention == 'patching':   # ZERO: no corrupted forward, so the delta is 0 - clean
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)   # activation_difference = +corrupted
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        extra = build_relp_fwd_hooks(model, use_mlp=use_mlp, use_qk=detach_qk, shapley_attn=shapley_attn,
                                     softmax_rule=softmax_rule, linearize_act=linearize_act,
                                     gim_attn=gim_attn) if relp_hooks else []
        with model.hooks(fwd_hooks=fwd_hooks_clean + extra, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)       # activation_difference -> corrupted - clean
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items
    return scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                              intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', steps=30, 
                              intervention_dataloader: Optional[DataLoader]=None, quiet:bool=False, neuron:bool=False):

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted + activations * 0
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                               quiet:bool=False, neuron:bool=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            logits = model(corrupted_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean'}      
def attribute_node(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                   method: Literal['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'exact'], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None,
                   quiet:bool=False, neuron:bool=False, optimal_ablation_path: Optional[str]=None,
                   mc_seed: int=0):
    # optimal_ablation_path is accepted for compatibility with MIB's run_attribution.py
    # (only the edge-level / 'optimal' ablation path uses it; ignored for these node methods).
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention, 
                                intervention_dataloader=intervention_dataloader, quiet=quiet, neuron=neuron)
    elif method == 'EAP-IG-inputs':
        if intervention not in ('patching', 'zero'):
            raise ValueError(f"intervention must be 'patching' or 'zero' for EAP-IG-inputs, but got {intervention}")
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet, neuron=neuron,
                                   intervention=intervention)
    elif method == 'EAP-IG-inputs-mc':
        if intervention not in ('patching', 'zero'):
            raise ValueError(f"intervention must be 'patching' or 'zero' for EAP-IG-inputs-mc, but got {intervention}")
        scores = get_scores_eap_ig_mc(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet,
                                      neuron=neuron, seed=mc_seed, intervention=intervention)
    elif method == 'EAP-IG-inputs-local':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs-local, but got {intervention}")
        scores = get_scores_eap_ig_local(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet, neuron=neuron)
    elif method == 'RelP':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron)
    elif method == 'RelP-qkgrad':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, detach_qk=False)
    elif method == 'RelPShapley':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, detach_qk=False, shapley_attn=True)
    elif method == 'AttnLRP':
        # Faithful AttnLRP: RelP's norm/identity/half rules + the uniform rule on both attention
        # matmuls, and the ORDINARY softmax gradient (AttnLRP Eq. 13 reduces to it; LXT patches
        # nothing there). This is RelPShapley minus the invented ShapleySoftmax rule.
        if intervention not in ('patching', 'zero'):
            raise ValueError(f"intervention must be 'patching' or 'zero' for AttnLRP, but got {intervention}")
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron,
                                 detach_qk=False, shapley_attn=True, softmax_rule=False,
                                 intervention=intervention)
    elif method == 'GIM':
        # GIM (Edin et al. 2026) = norm freeze + q/4,k/4,v/2 + tempered (T=2) softmax backward
        # + scale_mlp_gate. That last rule is `mlp_grad / 2` for GATED MLPs only, which is exactly
        # ShapleyElementwiseMult on the gate x up product: halving each branch's gradient and
        # summing them back at the MLP input gives (gate_grad + in_grad)/2 = mlp_grad/2. GIM keeps
        # the TRUE activation derivative though (activation_backward='grad'), so no secant
        # linearization -> linearize_act=False.
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, detach_qk=False,
                                 linearize_act=False, gim_attn=True)
    elif method in ('AtP', 'AtP-star'):
        # AtP = per-example |(noise - clean) . grad| (Kramar et al. 2024); AtP-star adds GradDrop.
        # The QK fix half of AtP* is vacuous on MIB's node set -- see get_scores_atp's docstring.
        if intervention not in ('patching', 'zero'):
            raise ValueError(f"intervention must be 'patching' or 'zero' for {method}, but got {intervention}")
        scores = get_scores_atp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron,
                                grad_drop=(method == 'AtP-star'), intervention=intervention)
    elif method == 'RelP-norules':
        scores = get_scores_relp(model, graph, dataloader, metric, quiet=quiet, neuron=neuron, relp_hooks=False)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, 
                                           intervention=intervention, intervention_dataloader=intervention_dataloader, 
                                           quiet=quiet, neuron=neuron)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, 
                                  intervention_dataloader=intervention_dataloader, 
                                  quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations'], but got {method}")


    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    if neuron:
        graph.neurons_scores[:] = scores.to(graph.scores.device)
    else:
        graph.nodes_scores[:] = scores.to(graph.scores.device)
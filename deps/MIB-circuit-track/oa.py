import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import os
import pickle
import argparse
from MIB_circuit_track.dataset import HFEAPDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from transformer_lens import HookedTransformer, HookedTransformerConfig
from torch.optim import Adam
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from MIB_circuit_track.UnifiedTransformer import UnifiedTransformer
from huggingface_hub import hf_hub_download


def load_interpbench_model():
    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits", filename="ll_model_100_100_80.pth")

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = "cuda"

    # Small hack to enable evaluation mode in the IOI model, that has a different config during training
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = UnifiedTransformer(cfg, state_dict=torch.load(hf_model, map_location="cuda"), from_cfg=True)
    return model


class OptimalAblationFinder:
    def __init__(
        self, 
        model: UnifiedTransformer,
        lr: float = 1e-3,
        num_steps: int = 1000,
        patience: int = 3,
        check_every: int = 100,
        target_components: List[str] = None,
        model_dtype = torch.float32
    ):
        """
        Initialize OptimalAblationFinder for Hugging Face models.
        
        Args:
            model: Hugging Face pre-trained model
            lr: Learning rate for optimization
            num_steps: Number of optimization steps
            target_components: List of component names to ablate. If None, defaults to MLPs and attention outputs
        """
        self.model = model
        self.lr = lr
        self.num_steps = num_steps
        self.patience = patience
        self.check_every = check_every
        self.dtype = model_dtype

        self.optimal_values = {}
        self.prev_best_loss = float('inf')
        self.not_improved = 0
        
        # If no target components specified, find all MLPs and attention blocks
        if target_components is None:
            self.target_components = self._find_default_components()
        else:
            self.target_components = target_components
            
    def _find_default_components(self) -> List[str]:
        """
        Automatically find MLP and attention output components in the model.
        
        Returns:
            List of component names
        """
        components = []
        
        def get_submodules(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this is an MLP or attention output
                if any(pattern in full_name.lower() for pattern in [
                    'hook_mlp_out', 
                    'attn.hook_result',
                    'hook_embed',
                    f'{model.cfg.n_layers-1}.hook_resid_post'
                ]):
                    components.append(full_name)
                
                # Recursively check children
                get_submodules(child, full_name)
        
        get_submodules(self.model)
        return components

    def compute_means(
        self,
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        mean_activations = {}

        # Initialize ablation values for each component
        for component in self.target_components:
            # Get the component's output size from a forward pass
            # layer = self.model.transformer.h[int(component.split('.')[2])]
            # submodule = eval(f"self.model.{component}")
            if 'attn' in component:  # attention
                zeros = torch.zeros(self.model.cfg.n_heads, self.model.cfg.d_model).to("cuda")
            else:
                zeros = torch.zeros(1, self.model.cfg.d_model).to("cuda")
                
            mean_activations[component] = zeros

        n_batches = 0
        for step in tqdm(range(self.num_steps), desc="Example:", total=(self.num_steps)):
            # Get batch
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                print("Encountered end of dataloader. Stopping...")
                break

            n_batches += 1
        
            # Move batch to model device
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            # Optimize each component's ablation value
            for component in self.target_components:
                submodule_vals = component.split(".")
                if len(submodule_vals) > 1:
                    layer = int(submodule_vals[1])
                else:
                    layer = None

                with torch.no_grad(): 
                    with self.model.trace(batch["input_ids"]) as handle:
                        # Set up hook for this component
                        if 'mlp' in component:
                            # target = eval(f"self.model.blocks[{layer}].mlp.hook_post")
                            target = self.model.blocks[layer].mlp
                        elif 'attn' in component:  # attention
                            # target = eval(f"self.model.blocks[{layer}].attn.hook_result")
                            target = self.model.blocks[layer].attn.hook_result
                        elif 'resid' in component:
                            target = self.model.blocks[layer].hook_resid_post
                        else:
                            # target = eval("self.model.hook_embed")
                            target = self.model.hook_embed
                        
                        batch_activations = target.output.save()

                    acts_agg = batch_activations.value.sum(dim=1).sum(dim=0).detach()
                    mean_activations[component] += acts_agg

        for component in self.target_components:
            mean_activations[component] /= n_batches
        return mean_activations
    
        
    def compute_optimal_ablation(
        self,
        dataloader: DataLoader,
        mean_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute optimal ablation values using dataloader.
        
        Args:
            dataloader: Dataloader containing inputs and labels
            
        Returns:
            Dictionary mapping component names to their optimal ablation values
        """
        # Initialize ablation values for each component
        ablation_values = {}
        optimizers = {}
        
        # Initialize session for getting output sizes
        # with self.model.generate() as sess:
            # outputs = self.model.forward(batch["input_ids"])
            
        # Initialize ablation values for each component to be their mean activations
        for component in self.target_components:
            ablation_values[component] = nn.Parameter(
                mean_activations[component],
                requires_grad=True
            )
            # Get the component's output size from a forward pass
            # layer = self.model.transformer.h[int(component.split('.')[2])]
            # submodule = eval(f"self.model.{component}")
            # if 'attn' in component:  # attention
            #     zeros = torch.zeros(self.model.cfg.n_heads, self.model.cfg.d_model)
            # else:
            #     zeros = torch.zeros(1, self.model.cfg.d_model)
                
            # ablation_values[component] = nn.Parameter(
            #     zeros,
            #     requires_grad=True
            # )
            optimizers[component] = Adam([ablation_values[component]], lr=self.lr)
        
        # Optimization loop
        for step in tqdm(range(self.num_steps), total=self.num_steps, desc="Step"):
            total_loss = 0
            
            # Get batch
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Move batch to model device
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            # Optimize each component's ablation value
            for component in self.target_components:
                submodule_vals = component.split(".")
                if len(submodule_vals) > 1:
                    layer = int(submodule_vals[1])
                else:
                    layer = None
                optimizers[component].zero_grad()
                
                with self.model.trace(batch["input_ids"]) as handle:
                    # Set up hook for this component
                    if 'mlp' in component:
                        # target = eval(f"self.model.blocks[{layer}].mlp.hook_post")
                        target = self.model.blocks[layer].mlp
                    elif 'attn' in component:  # attention
                        # target = eval(f"self.model.blocks[{layer}].attn.hook_result")
                        target = self.model.blocks[layer].attn.hook_result
                    elif 'resid' in component:
                        target = self.model.blocks[layer].hook_resid_post
                    else:
                        # target = eval("self.model.hook_embed")
                        target = self.model.hook_embed

                    if 'attn' in component:
                        ablated_output = ablation_values[component].expand(target.output.shape[0], target.output.shape[1],
                                                                            -1, -1).to(self.dtype)
                    else:
                        ablated_output = ablation_values[component].expand(target.output.shape[0],
                                                                           target.output.shape[1], -1).to(self.dtype)
                    target.output = ablated_output
                    # target.output = ablation_values[component].expand(target.output.shape[0], -1)
                        
                    # Register hook
                    # def forward_hook(module, input_, output):
                    #     return ablation_values[component].expand(output.shape[0], -1)
                    
                    # handle = target.register_forward_hook(forward_hook)

                    # OLD LOSS
                    # loss = self.model.output.sum().save()

                    logits = self.model.output.save()


                # Forward pass
                logits = logits.value
                shift_logits = logits[..., :-1, :].contiguous()
                labels = batch["input_ids"][..., 1:].contiguous()
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                # loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizers[component].step()
                total_loss += loss.item()
            
            if step % self.check_every == 0:
                print(f"Step {step}, Average Loss: {total_loss/len(self.target_components):.4f}")
                if total_loss/len(self.target_components) < self.prev_best_loss:
                    self.prev_best_loss = total_loss / len(self.target_components)
                    self.not_improved = 0
                else:
                    self.not_improved += 1
                    print(f"Loss not improved for {self.not_improved} evaluations (/ {self.patience})")
                    if self.not_improved >= self.patience:
                        break
        
        # Store final optimal values
        self.optimal_values = {
            component: value.detach()
            for component, value in ablation_values.items()
        }
        
        return self.optimal_values
    
    def apply_optimal_ablation(
        self, 
        component: str,
        permanent: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Apply the computed optimal ablation to a component.
        
        Args:
            component: Name of component to ablate
            permanent: If True, permanently modifies the model
            
        Returns:
            Original activation values if permanent=True, else None
        """
        if component not in self.optimal_values:
            raise ValueError(f"No optimal ablation value computed for {component}")
        
        module = self.model.get_submodule(component)
            
        def ablation_hook(module, input_, output):
            return self.optimal_values[component].expand(output.shape[0], -1)
        
        if permanent:
            # Store original parameters
            original_params = module.state_dict()
            
            # Replace with ablation values
            with torch.no_grad():
                for param in module.parameters():
                    param.copy_(self.optimal_values[component])
                    
            return original_params
        else:
            # Register hook for temporary ablation
            handle = module.register_forward_hook(ablation_hook)
            return handle

    def get_component_names(self) -> List[str]:
        """Return list of all target components."""
        return self.target_components


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="gpt2")
    parser.add_argument("--tasks", type=str, default="ioi")
    args = parser.parse_args()

    models = args.models.split(",") if "," in args.models else [args.models]
    tasks = args.tasks.split(",") if "," in args.tasks else [args.tasks]

    # Prepare dataset
    for model_name in models:
        for task in tasks:
            if task == "mcqa" and model_name == "gpt2":
                continue
            if task == "arc" and model_name in ("gpt2", "qwen2.5"):
                continue
            if os.path.exists(f"ablations/{model_name}/{task}_oa.pkl"):
                continue
            print(f"Evaluating {model_name} on {task}...")
            
            # Load model and tokenizer
            if model_name == "gpt2":
                model = UnifiedTransformer(model_name)
                model_dtype = torch.float32
            elif model_name == "interpbench":
                model = load_interpbench_model()
                model_dtype = torch.float32
            elif model_name == "qwen2.5":
                model = UnifiedTransformer("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
                model_dtype = torch.bfloat16
            else:
                model = UnifiedTransformer("google/gemma-2-2b", attn_implementation="eager")
                model_dtype = torch.bfloat16
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True if model_name in ("gemma2", "qwen2.5", "llama3") else False
            tokenizer = model.tokenizer

            split = "train"
            dataset_name = "mib-bench/copycolors_mcqa" if task == "mcqa" else "mib-bench/ioi"
            
            
            if task == "ioi":
                dataset = load_dataset(dataset_name, split="train")
            elif task == "mcqa":
                dataset = load_dataset(dataset_name, "4_answer_choices", split="train")
            tokenizer = model.tokenizer
            tokenizer.pad_token = tokenizer.eos_token

            # Create a custom dataset class
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.encodings = tokenizer(
                        texts,
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                        return_tensors="pt"
                    )

                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item['labels'] = item['input_ids'].clone()
                    return item

                def __len__(self):
                    return len(self.encodings.input_ids)

            # Create dataset
            tokenized_dataset = TextDataset(dataset['prompt'], tokenizer)

            # Data collator
            class DataCollator:
                def __call__(self, examples):
                    batch = {
                        'input_ids': torch.stack([example['input_ids'] for example in examples]),
                        'attention_mask': torch.stack([example['attention_mask'] for example in examples]),
                        'labels': torch.stack([example['labels'] for example in examples])
                    }
                    return batch

            # Create dataloader
            batch_size = 20 if model in ("gpt2", "qwen2.5", "interpbench") else 5
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                collate_fn=DataCollator(),
                shuffle=True
            )
            means_dataloader = deepcopy(dataloader)     # TODO: fix hack

            # Initialize and run
            finder = OptimalAblationFinder(model=model, num_steps=1000, check_every=10, patience=3,
                                           model_dtype=model_dtype)
            
            print("Computing mean activations...")
            mean_activations = finder.compute_means(means_dataloader)
            del means_dataloader                        # TODO: fix hack
            print("Done computing means!")
            print()

            os.makedirs(f"ablations/means/{model_name}", exist_ok=True)
            with open(f"ablations/means/{model_name}/{task}_oa.pkl", "wb") as handle:
                pickle.dump(mean_activations, handle)

            print("Optimizing over ablation vectors...")
            optimal_values = finder.compute_optimal_ablation(dataloader, mean_activations)
            print("Found optimal values! Saving...")
            print()
            
            os.makedirs(f"ablations/{model_name}", exist_ok=True)
            with open(f"ablations/{model_name}/{task}_oa.pkl", "wb") as handle:
                pickle.dump(optimal_values, handle)

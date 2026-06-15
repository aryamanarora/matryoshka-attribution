import torch

neurons = [1712,10297,8859,8887,3896,12728,2404,13492,8409,3644,2955,251,11029,10099,12154,11545,5700,3650,13003,12778,6339,8343,61,1448,1768,2372,6721,11096]

results = {}
for name, path in [("300", "plots/attribution_months_scores.pt"), ("3k", "plots/attribution_months_3k_scores.pt")]:
    d = torch.load(path, weights_only=False)
    scores = d["scores"]
    tokens = d["tokens"]
    total = scores.numel()
    flat = scores.flatten()
    ranks = flat.argsort(descending=True).argsort()
    r = {}
    for n in neurons:
        s = scores[18, :, n]
        best_pos = s.argmax().item()
        flat_idx = 18 * 13 * 14336 + best_pos * 14336 + n
        rank = ranks[flat_idx].item()
        pct = rank / total * 100
        r[n] = (s[best_pos].item(), best_pos, rank, pct, tokens[best_pos])
    results[name] = r

print(f"{'Neuron':>6s}  {'300step':>10s} {'Pos':>8s}  {'3kstep':>10s} {'Pos':>8s}  {'Delta':>8s}")
print("-" * 65)
for n in neurons:
    s3, p3, r3, pct3, t3 = results["300"][n]
    sk, pk, rk, pctk, tk = results["3k"][n]
    delta = pctk - pct3
    print(f"{n:>6d}  {pct3:>8.2f}% {t3:>8s}  {pctk:>8.2f}% {tk:>8s}  {delta:>+7.2f}%")

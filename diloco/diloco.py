global_model = GPT(cfg).cuda()
outer_opt = torch.optim.SGD(
    global_model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True
)
workers = [copy.deepcopy(global_model) for _ in range(k)]
inner_opts = [
    torch.optim.AdamW(w.parameters(), lr=inner_lr, weight_decay=0.1) for w in workers
]

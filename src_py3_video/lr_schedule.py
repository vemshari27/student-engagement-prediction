def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
	"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
	if iter_num%5000 == 0 and iter_num != 0:
		init_lr = init_lr*0.1
	else:
		init_lr = init_lr
    
	lr = init_lr * (1 + gamma * iter_num) ** (-power)

	i=0
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_lr[i]
		i+=1

	return optimizer


schedule_dict = {"inv":inv_lr_scheduler}

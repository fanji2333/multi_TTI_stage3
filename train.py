from algo.PPO_Lag import PPOLag
from algo.PPO_baseline import PPOBaseline
from algo.PPO import PPO
from algo.p3o import P3O
from common.tools import get_default_kwargs_yaml, plot


# if __name__ == "main":

cfgs = get_default_kwargs_yaml('P3O')
print('complete cfg load')

check_point = ''

agent = P3O(cfgs)

ep_ret, ep_cost, ep_len, path = agent.learn()

# ep_ret = [ele.cpu() for ele in ep_ret]
# ep_cost = [ele.cpu() for ele in ep_cost]
# ep_len = [ele.cpu() for ele in ep_len]

# cost_num = ep_cost[0].shape[0]
# ep_ave_cost = [[] for _ in range(cost_num)]
# for ep, ele in enumerate(ep_cost):
#     for idx in range(cost_num):
#         ep_ave_cost[idx].append(ele[idx].cpu() / ep_len[ep])

# ep_ave_ret = [ret/length if length != 0 else 0 for ret, length in zip(ep_ret, ep_len)]
# ep_ave_cost = [cost/length if length != 0 else 0 for cost, length in zip(ep_cost, ep_len)]
#
# ep_ave_ret = [ele.cpu() for ele in ep_ave_ret]
# ep_ave_cost = [ele.cpu().tolist() for ele in ep_ave_cost]
#
epochs = range(cfgs.train_cfgs.epochs)
plot(ep_ret, epochs, 'epoch ave return', path)
plot(ep_cost, epochs, 'epoch ave cost', path)

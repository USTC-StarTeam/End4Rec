# import mindspore as ms
# import mindspore.nn as nn
# from embedding import BehaviorAwareEmbedding
# from ebm_enhanced import EBMEnhanced
# # from hard_noise import HardNoiseEliminator
# # from soft_noise import SoftNoiseFilter

# import mindspore.ops as ops
# class SoftNoiseFilter(nn.Cell):
#     def __init__(self, d_model, num_behaviors):
#         super(SoftNoiseFilter, self).__init__()
#         self.num_behaviors = num_behaviors
#         self.linears = nn.CellList([nn.Dense(d_model, d_model) for _ in range(num_behaviors)])
#     def construct(self, S_hp, behavior_ids):
#         B, L, d = S_hp.shape
#         outputs_sp = []
#         outputs_sn = []
#         for i in range(B):
#             tokens = S_hp[i]
#             beh_ids = behavior_ids[i]
#             sp_list = []
#             for j in range(L):
#                 beh = int(beh_ids[j].asnumpy())
#                 token = tokens[j:j+1, :]
#                 sp_list.append(self.linears[beh](token))
#             sp = ops.Concat(axis=0)(sp_list)
#             sn = tokens - sp
#             outputs_sp.append(sp.expand_dims(0))
#             outputs_sn.append(sn.expand_dims(0))
#         S_sp = ops.Concat(axis=0)(outputs_sp)
#         S_sn = ops.Concat(axis=0)(outputs_sn)
#         return S_sp, S_sn

# class HardNoiseEliminator(nn.Cell):
#     def __init__(self, d_model, temperature=1.0):
#         super(HardNoiseEliminator, self).__init__()
#         self.fc = nn.Dense(d_model, 1)
#         self.sigmoid = ops.Sigmoid()
#         self.temperature = temperature
#     def construct(self, S):
#         t = self.fc(S)
#         p_b = ops.OnesLike()(t)
#         diff = p_b - t
#         prob = self.sigmoid(diff)
#         logits = ops.Concat(axis=-1)((prob, 1 - prob))
#         mask = ops.Greater()(prob, 0.5)
#         mask = ops.Cast()(mask, ms.float32)
#         S_hp = S * mask
#         S_hn = S * (1 - mask)
#         return S_hp, S_hn



# class END4Rec(nn.Cell):
#     def __init__(self, num_items, num_behaviors, seq_length, d_model, num_blocks=4, epsilon=1e-3):
#         super(END4Rec, self).__init__()
#         self.embedding = BehaviorAwareEmbedding(num_items, num_behaviors, seq_length, d_model)
#         self.ebm = EBMEnhanced(d_model, seq_length, num_blocks, epsilon)
#         self.hard_noise = HardNoiseEliminator(d_model)
#         self.soft_noise = SoftNoiseFilter(d_model, num_behaviors)
#         self.output_layer = nn.Dense(d_model, num_items)
#     def construct(self, item_ids, behavior_ids, positions):
#         x = self.embedding(item_ids, behavior_ids, positions)
#         S, reg_loss = self.ebm(x)
#         S_hp, S_hn = self.hard_noise(S)
#         S_sp, S_sn = self.soft_noise(S_hp, behavior_ids)
#         final_rep = S_sp[:, -1, :]
#         logits = self.output_layer(final_rep)
#         return logits, reg_loss, S, S_hp, S_hn, S_sp, S_sn

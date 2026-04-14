# class BlockMLP(nn.Cell):
#     def __init__(self, block_size):
#         super(BlockMLP, self).__init__()
#         self.fc1 = nn.Dense(block_size, block_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Dense(block_size, block_size)
#     def construct(self, x):
#         return self.fc2(self.relu(self.fc1(x)))
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
predict = np.array([[1, 2], [4, 5], [None, None]], np.float32)
target = np.array([[1, 2], [4, 5], [6, 7]], np.float32)
predict = torch.tensor(predict).to(device)
target = torch.tensor(target).to(device)
f_predict = predict[~torch.any(predict.isnan(),dim=1)]
f_target = target[~torch.any(predict.isnan(),dim=1)]

print(f_predict)
print(f_target)

#print(torch.any(tdata.isnan(),dim=1).nonzero(as_tuple=True)[0])
#print ((torch.any(tensor.isnan(),dim=1)).nonzero(as_tuple=True)[0])






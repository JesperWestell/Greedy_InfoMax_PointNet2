import torch

test = torch.Tensor([[[0,1,2],[10,11,12],[20,21,22]],
                     [[100,101,102],[110,111,112],[120,121,122]],
                     [[200,201,202],[210,211,212],[220,221,222]]])

#x,y,z

print(test)
print(test.size())

print(test[2,1,2])

print(test[:,0,0])

#print(test.reshape(27))

test2 = torch.Tensor([x*100 + y*10 + z for
                                  x in range(3)
                                  for y in range(3)
                                  for z in range(3)])

print(test2)
print(test2.reshape(3,3,3))

centers = torch.Tensor([[x, y, z] for
                                  x in range(3)
                                  for y in range(3)
                                  for z in range(3)])

print(centers.size())

centers = centers.reshape(3,3,3,-1)

print(centers[2,1,2])
print()

print(torch.Tensor([True, True, False]).mean())


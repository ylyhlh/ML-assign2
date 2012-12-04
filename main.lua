 --[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 10/10/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

-- Load required libraries and files
--dofile("spambase.lua")
dofile("isolet.lua")
dofile("whitening.lua")
require("nn")
dofile("MulPos.lua")
dofile("MulPosReg.lua")
dofile("NegExp.lua")
dofile("rbf.lua")
dofile("rbfa.lua")
dofile("models.lua")
dofile("linearReg.lua")

-- An example of using xsvm
function main() 
   local K=torch.Tensor({arg[1]})[1]--degreeiof kernel
   local H=torch.Tensor({arg[2]})[1]
   local n=torch.Tensor({arg[3]})[1]
   local lr=torch.Tensor({arg[4]})[1]/1000
   local Init_w=torch.Tensor({arg[5]})[1]
   local MaxIt=torch.Tensor({arg[6]})[1]
   local lmd=torch.Tensor({arg[7]})[1]/10000
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = isolet:getDatasets(n,1500)
   whitening:whitenDatasets(data_train, data_test,K)
   --local data_train, data_test = mnist:getDatasets(1000,1000)
   --mlp = modLogReg(data_train:features(),26)
   --mlp=nn.Linear(data_train:features(),26)
   --mlp = modTwoLay(data_train:features(),26,H)
  
   ----[[
   --mlp=modRBF(data_train:features(),H,26,Init_w)
   mlp=modRBFREG(data_train:features(),H,26,Init_w,lmd)
   criterion =nn.ClassNLLCriterion() -- Mean Squared Error criterion
   trainer = nn.StochasticGradient(mlp, criterion)
   print(lr)
   trainer.learningRate = lr
   --trainer.learningRateDecay=1
   trainer.shuffleIndices=1
   trainer.maxIteration = MaxIt
   trainer:train(data_train) -- train using some examples
   local error=modTest(mlp,data_test)
   print(error)
   local file=torch.DiskFile('rbf-'..K..'-'..H..'-'..Init_w..'-'..torch.Tensor({arg[4]})[1]..'.log','w')
   file:writeString(K..','..H..','..Init_w..','..n..','..lr..','..MaxIt..','..error..'\n')
   file:close()
   --]]
   --[[
   mlp=nn.RBF(3,3)
   ii=torch.ones(3)
   oo=mlp:forward(ii)
   go=torch.ones(3)
   print(mlp.weight)
   gi=mlp:backward(ii,go)
   print(mlp.weight)
   print()
   print(ii) print(oo) print(go) print(gi)
   mlp:updateParameters(1)
   print(mlp.weight)
   --mlp:zeroGradParameters()
   ii=torch.ones(3)
   oo=mlp:forward(ii)
   go=torch.ones(3)
   gi=mlp:backward(ii,go) 
   --mlp:updateParameters(1)
   print(mlp.weight)
   print(ii) print(oo) print(go) print(gi)
   print(mlp.weight)
   --]]--
end

main()

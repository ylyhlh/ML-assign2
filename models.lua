function modLogReg(inputs,outputs)
local mlp = nn.Sequential()
mlp:add( nn.Linear(inputs,outputs) ) 
mlp:add( nn.Tanh() ) 
mlp:add( nn.LogSoftMax() )
return mlp

end


function modTwoLayReg(inputs,outputs,hunits,lmd)
local mlp = nn.Sequential()
mlp:add( nn.LinearReg(inputs,hunits,lmd) )
mlp:add( nn.Tanh() ) 
mlp:add( nn.Linear(hunits,outputs,lmd) )
mlp:add( nn.LogSoftMax() )
return mlp
end

function modTwoLay(inputs,outputs,hunits)
local mlp = nn.Sequential()
mlp:add( nn.Linear(inputs,hunits) )
mlp:add( nn.Tanh() ) 
mlp:add( nn.Linear(hunits,outputs) )
mlp:add( nn.LogSoftMax() )
return mlp
end

function modRBF(inputs,hunits,outputs,initw)
local mlp = nn.Sequential()
----[[
--mlp:add( nn.LinearReg(inputs, inputs) )
--mlp:add( nn.Tanh() )
mlp:add( nn.RBF(inputs,hunits) )
mlp:add( nn.MulPos(hunits,initw))
mlp:add( nn.NegExp() ) 
--]]
mlp:add( nn.Linear(hunits,outputs) )
mlp:add( nn.LogSoftMax() )
return mlp

end
function modRBFREG(inputs,hunits,outputs,initw,lmd)
local mlp = nn.Sequential()
----[[
mlp:add( nn.RBF(inputs,hunits) )
mlp:add( nn.MulPosReg(hunits,initw))
mlp:add( nn.NegExp() ) 
--]]
--mlp:add( nn.LinearReg(hunits,hunits,lmd) )
--mlp:add( nn.Tanh() ) 
mlp:add( nn.LinearReg(hunits,outputs,lmd) )
mlp:add( nn.LogSoftMax() )
return mlp

end



function modTest(model, dataset)
   local error=0
for i=1,dataset:size() do
   local y=model:forward(dataset[i][1])
   local tmp,index=torch.max(y,1)
   if index[1]==dataset[i][2] then
      error = error/i*(i-1)
   else
      error = error/i*(i-1) + 1/i
   end
end

return error
end
----[[
function test_GrandInput(module,input)
 local small=1E-6
 local out_size=module:forward(input):nElement()
 local gradInput0= torch.Tensor():resize(input:nElement(),module:forward(input):nElement())
 local gradInput1= torch.Tensor():resize(input:nElement(),module:forward(input):nElement())
   for i=1,input:nElement() do      
      input[i] = input[i] + small
      local out0=module:forward(input):clone()
      input[i] = input[i] - 2*small
      local out1=module:forward(input):clone()
      input[i] = input[i] + small
      out0:add(-1,out1):div(2*small)
      gradInput1:select(1,i):copy(out0)
   end 
   for i=1,out_size do     
      local out=torch.zeros(out_size)
      out[i]=1
      local ingrad=module:backward(input,out):clone()
      gradInput0:select(2,i):copy(ingrad)
   end
   print('The GradInput error:'..torch.norm(gradInput0-gradInput1)) 
end
--]]--

function modLogReg(inputs,outputs)
local mlp = nn.Sequential()
mlp:add( nn.Linear(inputs,outputs) ) 
mlp:add( nn.Tanh() ) 
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
mlp:add( nn.RBFA(inputs,hunits) )
mlp:add( nn.MulPos(hunits,initw))
mlp:add( nn.NegExp() ) 
mlp:add( nn.Linear(hunits,outputs) )
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
--[[
function test_GrandInput(module,input)
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
   end  
end
]]--

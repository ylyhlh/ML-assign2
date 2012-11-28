local MulPos, parent = torch.class('nn.MulPos', 'nn.Module')

function MulPos:__init(inputSize,initw)
   parent.__init(self)

   self.weight = torch.Tensor({initw})
   self.gradWeight = torch.Tensor(1):zero()
   print(initw)
   -- state

end

 

function MulPos:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input)
   self.output:mul(torch.exp(self.weight[1]))
   --print(self.weight[1].."MUl"..torch.norm(self.output).."in"..torch.norm(input))
   return self.output 
end

function MulPos:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input)
   self.gradInput:ones(self.gradInput:size()):mul(torch.exp(self.weight[1])):cmul( gradOutput)
   return self.gradInput
end

function MulPos:accGradParameters(input, gradOutput, scale) 
   scale = scale or 1
      self.gradWeight[1] = self.gradWeight[1] + scale*torch.dot(gradOutput,input*torch.exp(self.weight[1]))
end

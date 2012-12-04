local MulPosReg, parent = torch.class('nn.MulPosReg', 'nn.Module')

function MulPosReg:__init(inputSize,initw)
   parent.__init(self)

   self.weight = torch.Tensor(inputSize):fill(initw)
   self.gradWeight = torch.Tensor(inputSize):zero()
   print(initw)
   -- state

end

 

function MulPosReg:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input)
   self.output:cmul(torch.exp(self.weight))
   --print(self.weight[1].."MUl"..torch.norm(self.output).."in"..torch.norm(input))
   return self.output 
end

function MulPosReg:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input)
   self.gradInput:ones(self.gradInput:size()):cmul(torch.exp(self.weight)):cmul( gradOutput)
   return self.gradInput
end

function MulPosReg:accGradParameters(input, gradOutput, scale) 
   scale = scale or 1
      self.gradWeight = self.gradWeight + scale*torch.dot(gradOutput,torch.cmul(input,torch.exp(self.weight)))
end

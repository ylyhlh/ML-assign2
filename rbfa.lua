local RBFA, parent = torch.class('nn.RBFA', 'nn.Module')

function RBFA:__init(inputSize,outputSize)
   parent.__init(self)

   self.templates = torch.rand(outputSize, inputSize)*2-1
   self.gradTemplates =torch.Tensor(outputSize, inputSize)

   self.covA = torch.Tensor(outputSize,inputSize):fill(1)
   self:normA()
   self.gradCovA = torch.Tensor(outputSize,inputSize)

   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)
   self.temp = torch.Tensor(inputSize)

   -- for compat with Torch's modules (it's bad we have to do that)
   do
      self.weight = self.templates
      self.gradWeight = self.gradTemplates
      self.bias = self.covA
      self.gradBias = self.gradCovA
   end

end

function RBFA:normA()
   for i=1,self.covA:size(1) do
      self.covA[i]:div(torch.norm(self.covA[i]))
   end
   
end
function RBFA:updateOutput(input)
   self.output:zero()
   self:normA()
   for o = 1,self.templates:size(1) do
      self.temp:copy(input):add(-1,self.templates[o])
      self.temp:cmul(self.temp)
      self.temp:cmul(self.covA:select(1,o))
      self.output[o] =self.temp:sum()
   end
   return self.output
end

function RBFA:updateGradInput(input, gradOutput)
   self.gradInput:zero()
   for o = 1,self.templates:size(1) do
         self.temp:copy(input*2):add(-2,self.templates:select(1,o))
         self.temp:cmul(self.covA:select(1,o))
         self.temp:mul(gradOutput[o])
         self.gradInput:add(self.temp)
   end
   return self.gradInput
end

function RBFA:accGradParameters(input, gradOutput, scale)
   self:forward(input)
   scale = scale or 1
   for o = 1,self.templates:size(1) do
         self.temp:copy(self.templates:select(1,o)*2):add(-2,input)
         self.temp:cmul(self.covA:select(1,o))
         self.temp:mul(gradOutput[o])
         self.gradTemplates:select(1,o):add(self.temp)

         self.temp:copy(self.templates:select(1,o)):add(-1,input)
         self.temp:cmul(self.temp)
         self.temp:mul(gradOutput[o])
         self.gradCovA:select(1,o):add(self.temp)
   end
end

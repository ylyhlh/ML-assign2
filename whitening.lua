whitening={};

function whitening:whitenDatasets(train,test,k)
  local X=torch.Tensor(train:features(),train:size()):zero()
  for i=1,train:size() do
    X[{{},i}]=train[i][1]
  end
  local Xt=torch.Tensor(test:features(),test:size()):zero()
  for i=1,test:size() do
    Xt[{{},i}]=test[i][1]
  end
  --using svd to compute the k mst-sig-eigvct of XX^*
  local u,s,v=torch.svd(X)
  --local u=torch.rand(train:features(),train:size())------------delete this line
  uk=u:sub(1,train:features(),1,k)
  uk=uk:t()
  --projecting  X onto k mst-sig-eigvct to get new features
  local Y=uk*X
  local Yt=uk*Xt
  --print(Y*Y:t())--this should be almost eye
  --refill train and test with new features
  for i=1,train:size() do
    train[i][1]=Y[{{},i}]
  end 
  for i=1,test:size() do
    test[i][1]=Yt[{{},i}]
  end
  --redefine the features() function
  function train:features() return k end
  function test:features() return k end
  --normalize train and test
  isolet:normalize(train,test)
  return train,test
end

require("nn")
dofile("models.lua")

function main()
mlp = nn.Sequential()
mlp:add(nn.Mul(10))
test_GrandInput(mlp,torch.rand(10))
end
main()

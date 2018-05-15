s = "aaabbb"

strs=["flower","flow","flight"]
l=len(strs[0])
for i in range(1,len(strs)):
    if l>len(strs[i]):
        l=len(strs[i])
s=""
for i in range(l):
    tmp = strs[0][i]
    for j in range(1,len(strs)):
        if strs[j][i]!=tmp:
            print(s)
            break
    s=s+tmp
print(s)
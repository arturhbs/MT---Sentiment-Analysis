

def sumArrayList(arr, sumArray):
    for i in range(len(arr)) :
        final = ''
        for w in arr[i]:
            final = final +' '
            final = final + w
        sumArray.append(final)


x = []
y = []
x.append('artur')
x.append('douglas')
y.append('rodrigo')
y.append('victor')
print("x=\n",x)
print("y=\n",y)
z=[]
z.append(x)
z.append(y)
print("z=\n",z)

z = x+y

print(z)
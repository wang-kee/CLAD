import seaborn as sns
import pandas
import matplotlib.pyplot as plt
 
csv = pandas.read_csv(r'results.csv')


xs = csv['Mixed-Same']
ys = csv['Mixed-Rand']
labels = csv['Model']
colors = csv['Color']
markers = []

for i in csv['Color']:
    if i == 'r':
        markers+=['o']
    else:
        markers+=['^']



from matplotlib import style
plt.style.use('ggplot')

# zip joins x and y coordinates in pairs
for x,y,label,c in zip(xs,ys,labels, colors):

    # label e

    if c == 'r':
        m = '*'
    elif c == 'b':
        m = '^'
    else:
        m = 'o'
    
    
    plt.scatter(x,y, c=c, marker=m)

    plt.xlim(60,92)
    plt.ylim(60,92)

    plt.xlabel('Mixed-Same')
    plt.ylabel('Mixed-Rand')


    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(8,2), # distance from text to points (x,y)
                 ha='left',
                 color=c,
                 fontsize=6,
                 ) # horizontal alignment can be left, right or center

xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
plt.savefig('results.png', dpi=300)
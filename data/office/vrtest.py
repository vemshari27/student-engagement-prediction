import os
from tqdm import tqdm


path = os.getcwd()


f = open(path+"/{}/{}_videodata_testlist.txt".format("VR","VR182"),"r")
f100 = open(path+"/{}/{}_100_videodata_testlist.txt".format("VR","VR182"),"w")
zero = 0
one = 0
cur = os.getcwd()
cur = cur.split('/')
cur.remove(cur[-1])
cur.remove(cur[-1])
f_path = "/".join(cur)
f_path = f_path+"/video_frames/{}/trimmed_video_frames".format("VR")

for line in f:
    line = line.split(' ')
    if(zero<20 and line[1][0]=='0'):
        zero=zero+1
        ans = "{}/{} 0\n".format(f_path,line[0].split('/')[-1])
    if(one<20 and line[1][0]=='1'):
        ans = "{}/{} 1\n".format(f_path,line[0].split('/')[-1])
        one+=1
    f100.write(ans)
    if(zero==20 and one==20):
        break
f.close()
f100.close()

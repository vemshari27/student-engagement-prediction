import os
from tqdm import tqdm


path = os.getcwd()


f = open(path+"/{}/{}_videodata_list.txt".format("POM","POM"),"r")
f100 = open(path+"/{}/{}_100_videodata_list.txt".format("POM","POM"),"w")
one = 0
zero = 0
cur = os.getcwd()
cur = cur.split('/')
cur.remove(cur[-1])
cur.remove(cur[-1])
f_path = "/".join(cur)
f_path = f_path+"/video_frames/{}/trimmed_video_frames".format("POM")
for line in f:
    line = line.split(' ')
    if(zero<50 and line[1][0]=='0'):
        zero=zero+1
        ans = "{}/{} 0\n".format(f_path,line[0].split('/')[-1])
    if(one<50 and line[1][0]=='1'):
        ans = "{}/{} 1\n".format(f_path,line[0].split('/')[-1])
        one+=1
    f100.write(ans)
    if(zero==50 and one==50):
        break;
f.close()
f100.close()

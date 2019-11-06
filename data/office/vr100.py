import os
from tqdm import tqdm


path = os.getcwd()


f = open(path+"/{}/{}_videodata_list.txt".format("VR","VR"),"r")
f100 = open(path+"/{}/{}_100_videodata_list.txt".format("VR","VR"),"w")
zero = 0
cur = os.getcwd()
cur = cur.split('/')
cur.remove(cur[-1])
cur.remove(cur[-1])
f_path = "/".join(cur)
f_path = f_path+"/video_frames/{}/trimmed_video_frames".format("VR")
for line in f:
    if(zero<100):
        zero=zero+1
        line = line.split('/')
        temp = line[-1].replace('\n','')
        ans = "{}/{}\n".format(f_path,temp)
        f100.write(ans)
    else:
        break
f.close()
f100.close()

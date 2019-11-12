import sys
import os
import shutil
n=int(sys.argv[1])

base=os.getcwd()

MultiplierDest=base+"/Multiplier_Dim%s"%n
if os.path.exists(MultiplierDest):
    pass
else:
    os.mkdir(MultiplierDest)

destSA=MultiplierDest+'/SA/'
destPT=MultiplierDest+'/PT/'

for path in [destSA,destPT]:
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

# =============================================================================
# 
# =============================================================================

PT_Source = base + "/DummyFiles/PT/"
SA_Source = base + "/DummyFiles/SA/"

AllContents_PT = os.listdir(PT_Source) 
AllContents_SA = os.listdir(SA_Source)

for filepath in AllContents_PT:
    contents = ''
    with open("DummyFiles/PT/"+filepath) as prop_con:
        for line in prop_con.readlines():
            contents+=line
    name = os.path.join(destPT,filepath)
    if os.path.exists(name):
        os.remove(name)
    f = open(name,"a+")
    f.write(contents.replace('Dimension',str(n)))
    f.close()

for filepath in AllContents_SA:
    contents = ''
    with open("DummyFiles/SA/"+filepath) as prop_con:
        for line in prop_con.readlines():
            contents+=line
    name = os.path.join(destSA,filepath)
    if os.path.exists(name):
        os.remove(name)
    f = open(name,"a+")
    f.write(contents.replace('Dimension',str(n)))
    f.close()
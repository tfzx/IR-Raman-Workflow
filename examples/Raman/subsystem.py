import dpdata

d = dpdata.System("./data", fmt = "deepmd/raw", type_map = ['O', 'H'])
d = d[0]
d.to("deepmd/raw", "./data2")
from string import maketrans

sentence = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."
a1 = "abcdefghijklmnopqrstuvwxyz"
a2 = "cdefghijklmnopqrstuvwxyzab"

trantab = maketrans(a1, a2)
print sentence.translate(trantab)
print "map".translate(trantab)

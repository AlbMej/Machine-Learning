PAD = '<$PAD$>'
SOS = '<$S$>'
EOS = '<$/S$>'
UNK = '<$UNK$>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

TOKEN_PADS = [
    (PAD, PAD_INDEX),
    (SOS, SOS_INDEX),
    (EOS, EOS_INDEX),
    (UNK, UNK_INDEX),
]

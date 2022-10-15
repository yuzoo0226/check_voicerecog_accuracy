###########################################
#                How 2 Use                #
###########################################

# GP_DICTIONARY
# dict_name : 任意のものを指定して良い，ただし他のものとかぶらないようにすること
#             音声認識関数の引数指定で，dictionary='dict_name'となるように指定する
# example: voicerecog(language="ja", dictionary="name", is_use_unknown=True)

GP_DICTIONARY = {
    # "dict_name" ["sample", "example", "test"], # dict_nameという名前で定義された[sample, example, test]の3単語を認識できる辞書
    "name": ["alex", "alice", "blair", "charlie", "charlotte", "chloe", "emery", "emma", "harper", "olivia", "sophia",
             "aiden", "aubrey", "brook", "carter", "dakota", "finley", "leonardo", "liam", "lucas", "noah", "oliver"
             ],
    "drink": ["coke", "water"],
    "num_ja": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
}

# 置換辞書
GP_RPLACE_DIC = {
    "lemon_lime": ["lemon [unk]", "[unk] lime", "lemon lime"],
    "aloe_drink": ["aloe [unk]", "aloe drink"]
}
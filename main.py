import textwrap

# parity bit drop for 64-bit key
keyp = [57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 27, 19, 11, 3,
        60, 52, 44, 36, 63, 55, 47, 39,
        31, 23, 15, 7, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 28, 20, 12, 4]

# Permutation table for key for first round
key_comp = [14, 17, 11, 24, 1, 5, 3, 28,
            15, 6, 21, 10, 23, 19, 12, 4,
            26, 8, 16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55, 30, 40,
            51, 45, 33, 48, 44, 49, 39, 56,
            34, 53, 46, 42, 50, 36, 29, 32]

#shift table we only use index 0
shift_table = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# Initial Permutation Table
initial_perm = [58, 50, 42, 34, 26, 18, 10, 1,
                60, 52, 44, 36, 28, 20, 12, 4,
                62, 54, 46, 38, 30, 22, 14, 6,
                64, 56, 48, 40, 32, 24, 16, 8,
                57, 49, 41, 33, 25, 17, 9, 2,
                59, 51, 43, 35, 27, 19, 11, 3,
                61, 53, 45, 37, 29, 21, 13, 5,
                63, 55, 47, 39, 31, 23, 15, 7]

# Final Permutation Table
final_perm = [8, 40, 48, 16, 56, 24, 64, 32,
              39, 7, 47, 15, 55, 23, 63, 31,
              38, 6, 46, 14, 54, 22, 62, 30,
              37, 5, 45, 13, 53, 21, 61, 29,
              36, 4, 44, 12, 52, 20, 60, 28,
              35, 3, 43, 11, 51, 19, 59, 27,
              34, 2, 42, 10, 50, 18, 58, 26,
              33, 1, 41, 9, 49, 17, 57, 25]

expansion_table = [32, 1, 2, 3, 4, 5,
                   4, 5, 6, 7, 8, 9,
                   8, 9, 10, 11, 12, 13,
                   12, 13, 14, 15, 16, 17,
                   16, 17, 18, 19, 20, 21,
                   20, 21, 22, 23, 24, 25,
                   24, 25, 26, 27, 28, 29,
                   28, 29, 30, 31, 32, 1]

sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
        ]


def generate_round_key(key):
    # Perform initial permutation on key

    bitDrop = []
    for bit in keyp:
        bitDrop.append(key[bit-1])

    left = bitDrop[:28]
    right = bitDrop[28:]

    shift_value = shift_table[0]
    left = left[shift_value:] + left[:shift_value]
    right = right[shift_value:] + right[:shift_value]
    concatKey = left + right

    round1_key = [concatKey[bit - 1] for bit in key_comp]
    return round1_key


def pad_text(text):
    padding_length = 8 - (len(text) % 8)
    padding = chr(padding_length) * padding_length
    return text + padding


def invfinalperm():
    inverse_final_perm = [final_perm.index(i) + 1 for i in range(1, 65)]
    return inverse_final_perm


def expansion_permutation(input_bits):
    output_bits = []
    for i in range(48):
        output_bits.append(input_bits[expansion_table[i] - 1])
    return output_bits


def xor_lists(list1, list2):
    result = []
    for a, b in zip(list1, list2):
        if a == b:
            result.append('0')
        else:
            result.append('1')
    return result


def s_box(input):
    # Divide the input into 8 blocks of 6 bits each
    blocks = [input[i:i+6] for i in range(0, 48, 6)]
    result = []

    for i in range(8):
        block = blocks[i]
        row = int(block[0] + block[5], 2)
        column = int(block[1]+block[2]+block[3]+block[4], 2)
        value = sbox[i][row][column]
        result.extend([int(b) for b in format(value, '04b')])

    return result


def convertToBinaryP(text):
    bin = ''.join(format(ord(i), '08b') for i in text)
    return bin


def convertToBinaryC(Hex):
    bin = ''.join(['{0:04b}'.format(int(d, 16)) for d in Hex])
    return bin


def find_p_box(inputs, outputs):

    list = [[[]]]
    list_indexs = []
    for i in range(len(outputs)):
        out = outputs[i]
        tempList = []
        listZero = []
        listOne = []
        for j in range(len(out)):
            if inputs[i][j] == '0':
                listZero.append(j)
            else:
                listOne.append(j)
        for bit in range(len(out)):
            if out[bit] == '0':
                tempList.append(listZero)
            else:
                tempList.append(listOne)
        list_indexs.append(tempList)

    for item in list_indexs:
        print(item)

    listSub = list_indexs[0]

    for i in range(len(list_indexs)-1):
        ll = []
        for j in range(len(list_indexs[i+1])):
            sub = []
            for element in set(listSub[j]):
                if element in set(list_indexs[i+1][j]):
                    sub.append(element)
            ll.append(sub)

        listSub = ll

    print(listSub)
    return listSub

def getInputs(key):

    inverse_final_perm = invfinalperm()
    list = [["kootahe", "6E2F7B25307C3144"],
             ["Zendegi", "CF646E7170632D45"],
             ["Edame", "D070257820560746"],
             ["Dare", "5574223505051150"],
             ["JolotYe", "DB2E393F61586144"],
             ["Daame", "D175257820560746"],
             ["DaemKe", "D135603D1A705746"],
             ["Mioftan", "D83C6F7321752A54"],
             ["Toosh", "413A2B666D024747"],
             ["HattaMo", "5974216034186B44"],
             ["khayeSa", "EA29302D74463545"],
             ["05753jj", "B1203330722B7A04"],
             ["==j95697","38693B6824232D231D1C0D0C4959590D"]
            ]
    removeIndex = []
    counter = 0
    for item in list:
        if len(item[1]) > 16:
            removeIndex.append(counter)
            blocksPlain = textwrap.wrap(item[0], 8)
            blocksCipher = textwrap.wrap(item[1], 16)
            num = len(blocksCipher)
            for i in range(num):
                p = ''
                if len(blocksPlain) > i:
                    p = blocksPlain[i]
                c = blocksCipher[i]
                test = [p, c]
                list.append(test)
            # print(list)
    for remove in removeIndex:
        list.pop(remove)
    #convert list to binary form
    binarylist = []
    for item in list:
        if len(item[0]) < 8:
            plainPad = pad_text(item[0])
        elif len(item[0]) >= 8:
            item[0] = item[0][:8]
            plainPad = item[0]

        binP = convertToBinaryP(plainPad)
        binC = convertToBinaryC(item[1])
        tempList = [binP, binC]
        binarylist.append(tempList)
        counter += 1
    # print(len(list))
    # print(len(binarylist))
    # for remove in removeIndex:
    #     binarylist.pop(remove)
    # print(len(binarylist))


    listPairs = []
    inputs = []
    outputs = []
    for item in binarylist:
        # permutate plain text and encrypt it (all steps before straight p-box)
        plain_text_per = []
        for bit in initial_perm:
            plain_text_per.append(item[0][bit - 1])

        result_encrypt = encrypt(plain_text_per[32:], key)

        # permutate cipher text with inverse of final permutate
        cipher_text_inverse = []
        for bit in inverse_final_perm:
            cipher_text_inverse.append(item[1][bit - 1])
        result_dycript = reverse(cipher_text_inverse, plain_text_per)
        stringE = ''.join(chr(i + 48) for i in result_encrypt)
        stringD = ''.join(result_dycript)
        temp = [stringE, stringD]
        print(temp)

        print("---------------------")
        listPairs.append(temp)
        inputs.append(stringE)
        outputs.append(stringD)

    p_box = find_p_box(inputs, outputs)
    print(p_box)
    return p_box

def showHex(text):
    binary_strp = ''.join(str(bit) for bit in text)
    hex_strp = hex(int(binary_strp, 2))[2:]
    return hex_strp

def encrypt(plainRight, key):
    exp_perm = expansion_permutation(plainRight)
    resultXor = xor_lists(key, exp_perm)
    return s_box(resultXor)


def reverse(cipherText, plainText):
    plainLeft = plainText[:32]
    cipherLeft = cipherText[:32]
    resultXor = xor_lists(plainLeft, cipherLeft)
    return resultXor

def findPlain(pbox, cipher, key):
    blocks = textwrap.wrap(cipher, 16)
    result = ''
    for i in range(len(blocks)):
        bin = convertToBinaryC(blocks[i])
        initialPer = []
        for bit in initial_perm:
            initialPer.append(bin[bit - 1])
        rightPart = initialPer[32:]
        resultSbox = encrypt(rightPart, key)
        straight = []
        for p in pbox:
            item = p[0]
            straight.append(resultSbox[item])
        output_list = [str(bit) for bit in straight]
        leftPart = initialPer[:32]
        xorLeftF = xor_lists(output_list, leftPart)

        concatLR = xorLeftF + rightPart
        plainBin = []
        for bit in final_perm:
            plainBin.append(concatLR[bit - 1])
        string = ''
        for i in plainBin:
            string = string+i
        binary_string = ''.join(plainBin)
        hex = showHex(plainBin)
        if len(hex) < 16:
            num = 16 - len(hex)
            zero = ''
            for k in range(num):
                zero = zero + '0'
            hex = zero + hex

        result = result + (bytes.fromhex(hex).decode('utf-8'))

    last_byte = result[-1]
    padding_length = ord(last_byte)
    result = result[:-padding_length]
    print(result)


# initial key value
key = '4355262724562343'
givenCipher = '59346E29456A723B62354B61756D44257871650320277C741D1C0D0C4959590D'

# convert key value to binary
key_binary = bin(int(key, 16))[2:].zfill(64)

# generate round 1 key
round1_key = generate_round_key(list(key_binary))  # Generate round keys

straightPbox = getInputs(round1_key)
findPlain(straightPbox, givenCipher, round1_key)

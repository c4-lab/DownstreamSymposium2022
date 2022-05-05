def cleanProp(row):
    if len(row.split(' : ')) == 2:
        id, sentence = row.split(' : ')
        sentence = sentence.replace('-----', ' ').replace('\n', '')
        return(id, sentence)
    else:
        return(None)
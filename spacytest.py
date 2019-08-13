import string
import spacy

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    
    with open('./data/onephrase_north_clean_src.txt', "r") as f:
        lines = f.readlines()
    
    out = open("./data/onephrase_north_clean_underscored_src.txt", "w") 

    for line in lines:
        line = line.replace('\n', '')
        #line = "".join(c for c in line if c not in string.punctuation)

        doc = nlp(line)
        for chunk in doc.noun_chunks:
            obj = chunk.text
            if obj.startswith('the '):
                obj = obj[4:]
            line = line.replace(obj, obj.replace(' ', '_'))
        out.write(line + '\n')

    out.close()

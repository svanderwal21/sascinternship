def read():
    conv_dic = {}
    biomart = "mart_export.txt"
    with open(biomart, "r") as mart:
        for line in mart:
            line = line.split(",")
            conv_dic[line[1].strip("\n")] = line[0].strip("\n")
    #print(conv_dic)
    
    new_dic = {}
    wikipathways= "wikipathways.gmt"
    with open(wikipathways, "r") as wiki:
        for line in wiki.readlines():
            new_line = []
            line = line.split("\t")
            pathway = line[0].split("%")[0]
            #print(pathway)
            for gene in line[2:]:
                gene = gene.strip("\n")
                conv_gene = conv_dic.get(gene)
                if conv_gene == '':
                    conv_gene = gene
                new_line.append(conv_gene)
            new_dic[pathway] = new_line

    return new_dic
            
                
def write(new_dic):
    file = open("custom_traintest.txt", "w")
    for x in new_dic:
        file.write(x+"\t")
        for y in new_dic[x]:
            file.write(str(y)+"\t")
        file.write("\n")
    file.close()
        
    




def main():
    new_dic = read()
    #make like 80/20 split
    write(new_dic)



main()

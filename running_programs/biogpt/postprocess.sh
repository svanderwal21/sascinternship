cat biogpt.txt | sed 's/, /\n/g' | grep -v 'FREETEXT\|PARAGRAPH\|Click here to view' > biogpt.txt 

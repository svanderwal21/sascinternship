cat biomed.txt | sed 's/, /\n/g' | grep -v 'FREETEXT\|PARAGRAPH\|Click here to view' > biomed.txt 

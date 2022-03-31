import pandas as pd
def main():
    #write dev.pos/dev.neg and attr
    ft= open("dev_30.txt",'r')
    fa= open("dev_30.attr",'r')
    ft_pos=open('dev.pos','w')
    ft_neg=open('dev.neg','w')
    lines=ft.readlines()
    labels=fa.readlines()
    for line,lab in zip(lines,labels):
        if lab[0]=='n':
            ft_neg.write(line.strip()+'\n')
            # print(line)
            # print(lab)
        else:
            ft_pos.write(line.strip()+'\n')
  
    ft.close()
    fa.close()
    ft_pos.close()
    ft_neg.close()


    #write in csv file,column=text
    pos_ft=open('dev.pos','r')
    neg_ft=open('dev.neg','r')
    pos_new_lines=[]
    pos_lines=pos_ft.readlines()
    for line in pos_lines:
        new_line=line.strip()
        pos_new_lines.append(new_line)
    neg_new_lines=[]
    neg_lines=neg_ft.readlines()
    for line in neg_lines:
        new_line=line.strip()
        neg_new_lines.append(new_line)

    pos_csv=pd.DataFrame(pos_new_lines,columns=['text'])
    neg_csv=pd.DataFrame(neg_new_lines,columns=['text'])
    pos_csv.to_csv('dev.pos.csv',index=False)
    neg_csv.to_csv('dev.neg.csv',index=False)
    pos_ft.close()
    neg_ft.close()
    print('write in csv')

    #write train part
    ft= open("train_30.txt",'r')
    fa= open("train_30.attr",'r')
    ft_pos=open('train.pos','w')
    ft_neg=open('train.neg','w')
    lines=ft.readlines()
    labels=fa.readlines()
    for line,lab in zip(lines,labels):
        if lab[0]=='n':
            ft_neg.write(line.strip()+'\n')
        else:
            ft_pos.write(line.strip()+'\n')
    ft.close()
    fa.close()
    ft_pos.close()
    ft_neg.close()

    #write in csv file,column=text
    pos_ft=open('train.pos','r')
    neg_ft=open('train.neg','r')
    pos_new_lines=[]
    pos_lines=pos_ft.readlines()
    for line in pos_lines:
        new_line=line.strip()
        pos_new_lines.append(new_line)
    neg_new_lines=[]
    neg_lines=neg_ft.readlines()
    for line in neg_lines:
        new_line=line.strip()
        neg_new_lines.append(new_line)

    pos_csv=pd.DataFrame(pos_new_lines,columns=['text'])
    neg_csv=pd.DataFrame(neg_new_lines,columns=['text'])
    pos_csv.to_csv('train.pos.csv',index=False)
    neg_csv.to_csv('train.neg.csv',index=False)
    pos_ft.close()
    neg_ft.close()
    print('write in csv')
    
    #write test part
    ft= open("test_30.txt",'r')
    fa= open("test_30.attr",'r')
    ft_pos=open('test.pos','w')
    ft_neg=open('test.neg','w')
    lines=ft.readlines()
    labels=fa.readlines()
    for line,lab in zip(lines,labels):
        if lab[0]=='n':
            ft_neg.write(line.strip()+'\n')
        else:
            ft_pos.write(line.strip()+'\n')
    ft.close()
    fa.close()
    ft_pos.close()
    ft_neg.close()

    #write in csv file,column=text
    pos_ft=open('test.pos','r')
    neg_ft=open('test.neg','r')
    pos_new_lines=[]
    pos_lines=pos_ft.readlines()
    for line in pos_lines:
        new_line=line.strip()
        pos_new_lines.append(new_line)
    neg_new_lines=[]
    neg_lines=neg_ft.readlines()
    for line in neg_lines:
        new_line=line.strip()
        neg_new_lines.append(new_line)

    pos_csv=pd.DataFrame(pos_new_lines,columns=['text'])
    neg_csv=pd.DataFrame(neg_new_lines,columns=['text'])
    pos_csv.to_csv('test.pos.csv',index=False)
    neg_csv.to_csv('test.neg.csv',index=False)
    pos_ft.close()
    neg_ft.close()
    print('write in csv')


if __name__ == '__main__':
    main()

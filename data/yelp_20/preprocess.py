def main():
    ft= open("dev_20.txt",'r')
    fa= open("dev_20.attr",'r')
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

    ft= open("train_20.txt",'r')
    fa= open("train_20.attr",'r')
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

    ft= open("test_20.txt",'r')
    fa= open("test_20.attr",'r')
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

if __name__ == '__main__':
    main()
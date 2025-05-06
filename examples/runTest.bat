..\\GRNTIClass\\GRNTIClass.exe models config
..\\GRNTIClass\\GRNTIClass.exe predict -i text.txt -o resSimple.csv -id RGNTI2 -f plain -l ru -t 0.1
..\\GRNTIClass\\GRNTIClass.exe predict -i test_ru.csv -o resMultidoc.csv -id RGNTI2 -f multidoc -l ru -t 0.1
..\\GRNTIClass\\GRNTIClass.exe predict -i test_ru_500.csv -o resMultidoc500.csv -id RGNTI2 -f multidoc -l ru -t 0.1
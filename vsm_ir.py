import sys
from retrieve_doc import retrieval
from Inverted_Index import create_index


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("missing arguments")
        exit(-1)
    if args[1] == 'create_index':
        if len(args) <= 2:
            print("missing corpus directory")
            exit(-1)
        corpus_dir = args[2]
        create_index(corpus_dir)
    elif args[1] == 'query':
        if len(args) <= 3:
            print("missing index path and/or query")
            exit(-1)
        index_path,query = args[2],args[3]
        res = retrieval(index_path,query)
        with open("ranked_query_docs.txt",'w+') as of:
            num_of_doc = 0
            for doc in res.keys():
                num_of_doc += 1
                of.write(f"{doc}\n")
                if num_of_doc == 29:
                    break
    else:
        print(f"wrong command provided - '{args[1]}'")
        exit(-1)

    exit(0)
        
    
import argparse
import os
import json 
import torch

from src.empathy_classifier import EmpathyClassifier


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--er_path', default='empathy-models/empathy-emo-reactions-classifer.pth')
    parser.add_argument('--ip_path',default='empathy-models/empathy-interpretations-classifer.pth')
    parser.add_argument('--ex_path',default='empathy-models/empathy-explorations-classifer.pth')
    parser.add_argument('--seeker_post', help='insert seeker post', default='I am lost about life')
    parser.add_argument('--response_post', help='insert response post', default='thats too bad!')
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    
    args=parser.parse_args()

    # load model
    empathy_classifier=EmpathyClassifier(
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        ER_model_path=args.er_path,
        IP_model_path=args.ip_path,
        EX_model_path=args.ex_path
        )

    while True:
        # store seeker and response post in a list
        seeker_post=input('input seeker post: \n')
        if seeker_post == 'exit':
            exit()
        response_post=input('input response post: \n')

        
        seekerRespPostPair=[seeker_post, response_post]
        
        # store posts and outputs in a dict 
        results_dict={
            'seeker_post': args.seeker_post,
            'response_post': args.response_post
        }
        
        # predict: first argument is for printing purposes, so i'm leaving it out
        results_str, er_score, ex_score, ip_score=empathy_classifier.compute_empathy(seekerRespPostPair)
        
        # scores are log probability over multinomial random variables with k=3 [0,1,2]. 
        # argmax to obtain top value
        results_dict['ER']=torch.argmax(er_score).item()
        results_dict['EX']=torch.argmax(ex_score).item()
        results_dict['IP']=torch.argmax(ip_score).item()
        
        print(results_str)
        if args.save:
            # save the example as a json file 
            with open(f"saved_examples/example_{len(os.listdir('saved_examples'))+1}.json",'w') as f:
                json.dump(results_dict,f,indent=2)

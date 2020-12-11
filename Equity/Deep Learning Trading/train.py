import tensorflow as tf
import numpy as np
import random
import convNN as CNN
import exReplay as exR
import os
import pandas as pd


gpu_config = tf.compat.v1.ConfigProto()  
gpu_config.gpu_options.allow_growth = True # only use required resou)àrce(memory)
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # restrict to 50%


class trainModel:

    def __init__(self, epsilon_init, epsilon_min, maxiter, Beta, B,C, learning_rate, P):

        self.DataX          = list() 
        self.DataY          = list()

        self.epsilon        = epsilon_init 
        self.epsilon_min    = epsilon_min

        self.maxiter        = maxiter
        self.Beta           = Beta
        self.learning_rate  = learning_rate 
        self.P              = P 
        self.B              = B
        self.C              = C

    def set_Data(self, DataX, DataY):
        self.DataX = DataX
        self.DataY = DataY

        print ('X Data:  Comp#, Weeks# ', len(self.DataX), len(self.DataX[0]))
        print ('Y Data:  Comp#, Weeks# ', len(self.DataY), len(self.DataY[0]))

    def trainModel (self, H,W, FSize, PSize, PStride, NumAction, M, Gamma):

        # place holder
        state       = tf.placeholder (tf.float32, [None,H,W])
        isTrain     = tf.placeholder (tf.bool, [])

        Action      = tf.placeholder (tf.float32, [None,NumAction])
        Target      = tf.placeholder (tf.float32, [None,NumAction])

        # construct Graph
        C           = CNN.ConstructCNN(H,W, FSize, PSize, PStride, NumAction)
        rho_eta     = C.QValue    (state, isTrain)
        Loss_Tuple  = C.optimize_Q(rho_eta[0], Action, Target, self.Beta, self.learning_rate)

        sess        = tf.Session (config = gpu_config)    # maintains network parameter theta
        sessT       = tf.Session (config = gpu_config)    # maintains target network parameter theta^*
        sess.run (tf.global_variables_initializer())

        # saver
        saver       = tf.train.Saver(max_to_keep = 20)

        # copy inital
        saver.save      (sess, 'DeepQ')
        saver.restore   (sess, 'DeepQ')
        saver.restore   (sessT, 'DeepQ')

        # current experience
        preS    = np.empty((1,H,W), dtype = np.float32)
        preA    = np.empty((NumAction), dtype = np.int32)

        curS    = np.empty((1,H,W), dtype = np.float32)
        curA    = np.empty((NumAction), dtype = np.int32)
        curR    = 0
        nxtS    = np.empty((H,W), dtype = np.float32)

        memory  = exR.exRep(M, W, H)  # memory buffer
        b       = 1                     # iteration counter
        
        company = 0
        day = 1
        N = len(self.DataX)
        days = len(self.DataX[0]) -1
        while True:

            #1.0 get random valid index c, t
            if day >= days:
                day = 1
                t = day
                if company>=N:
                    day+=1
                    company=0
                    c = company
                    company+=1
                else:
                    c = company
                    company+=1
            else:
                t = day
                if company>=N:
                    day+=1
                    company=0
                    c = company
                    company+=1
                else:
                    c = company
                    company+=1
            # #1.0 get random valid index c, t
            # c       = random.randrange(0, len(self.DataX))
            # t       = random.randrange(1, len(self.DataX[c]) -1)
            
            #1.1 get preS
            preS    = self.DataX[c][t-1]
            
            #1.2 get preA by applying epsilon greedy policy to preS
            if(self.randf(0,1) <= self.epsilon):
                preA        = self.get_randaction   (NumAction) 
            else:                    
                QAValues    = sess.run              (rho_eta, feed_dict={ state: preS.reshape(1,H,W), isTrain:False })
                preA        = QAValues[1].reshape   (NumAction)

            #1.3 get curS
            curS    = self.DataX[c][t]

            #1.4 get curA by applying epsilon greedy policy to curS
            if( self.randf(0,1) <= self.epsilon):
                curA        = self.get_randaction   (NumAction) 
            else:                    
                QAValues    = sess.run              (rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False })
                curA        = QAValues[1].reshape   (NumAction)

            #1.5 get current reward and next state
            curR    = self.get_reward(preA, curA, self.DataY[c][t], self.P)
            nxtS    = self.DataX[c][t+1]

            #1.6 remember experience : tuple of curS, curA, curR, nxtS   
            memory.remember(curS, curA, curR, nxtS)

            #1.7: set epsilon                       
            if (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon * 0.999999  

            #2: update network parameter theta  every B iteration
            if (len(memory.curS) >= M) and (b % self.B == 0) :

                #2.1:  update Target network parameter theta^*
                if(b % (self.C * self.B) == 0)  : 
                    saver.save(sess, 'DeepQ')
                    saver.restore(sessT, 'DeepQ')

                #2.2: sample Beta size batch from memory buffer and take gradient step with respect to network parameter theta 
                S,A,Y   = memory.get_Batch  (sessT, rho_eta, state, isTrain,  self.Beta, NumAction, Gamma)
                Opts    = sess.run          (Loss_Tuple, feed_dict = { state:S, isTrain:True, Action:A, Target:Y }  )

                #2.3: print Loss 
                if( b % ( 100 * self.B  ) == 0 ):
                    print ('Loss: ' ,b, Opts[0])

            #3: update iteration counter
            b   = b + 1

            #4: save model 
            if( b >= self.maxiter ):
                saver.save( sess, 'DeepQ' )
                sess.close()
                sessT.close()
                print ('Finish! ')
                return 0
    
    def TestModel_ConstructGraph    ( self, H,W, FSize, PSize, PStride,  NumAction  ):

        # place holder
        state       = tf.placeholder (tf.float32, [None,H,W])
        isTrain     = tf.placeholder (tf.bool, [])

        #print tf.shape( isTrain)
        #print(tf.__version__)

        # construct Graph
        C           = CNN.ConstructCNN(H,W, FSize, PSize, PStride, NumAction)
        rho_eta     = C.QValue(state, isTrain)

        sess        = tf.Session (config = gpu_config)
        saver       = tf.train.Saver()

        return sess, saver, state, isTrain, rho_eta


    def validate_Neutralized_Portfolio(self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W):
       
        # list
        N           = len(DataX)
        Days        = len(DataX[0])
        curA        = np.zeros((N, NumAction))

        # alpha
        preAlpha_n  = np.zeros(N)
        curAlpha_n  = np.zeros(N)
        posChange   = 0

        # reward
        curR        = np.zeros(N)
        avgDailyR   = np.zeros(Days)


        # cumulative asset:  initialize cumAsset to 1.0
        cumAsset  = 1
        
        count = 0
        
        goodpreds = 0
        badpreds = 0
        totalpreds = 0
        
        posgoodpreds = 0
        posbadpreds = 0
        postotalpreds = 0
        
        neggoodpreds = 0
        negbadpreds = 0
        negtotalpreds = 0
        
        array_companies = []
        array_days = []
        
        for i in range(N):
            array_companies+=[i]*(Days - 1)
            array_days+=[j for j in range(Days - 1)]
        arrays = [array_companies, array_days]
        
        index = pd.MultiIndex.from_arrays(arrays, names=('Company', 'Date'))
        columns = ['Predicted Signal', 'Real Signal', 'Quantity', 'Tomorrow Returns', 'Reward', 'Cumulative Reward at t']
        df_results = pd.DataFrame(index=index, columns = columns)

        for t in range(Days - 1):
            for c in range(N):
           
                #1: choose action from current state 
                curS        = DataX[c][t]
                QAValues    = sess.run(rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False })
                curA[c]     = np.round(QAValues[1].reshape((NumAction)))
            
            # set Neutralized portfolio for day t
            curAlpha_n  = self.get_NeutralizedPortfolio(curA,  N)
            
            alpha = np.zeros(N)
            # Get Alpha (Signal -1, 0 or 1)
            for c in range (N):
                alpha[c]    = 1 - np.argmax(curA[c])
            curAlpha = alpha

            for c in range(N) :
                
                if DataY[c][t]>0:
                    if curAlpha[c]>=0:
                        count+=1
                    if curAlpha[c] != 0:
                        if curAlpha[c]>0:
                            goodpreds+=1
                            totalpreds+=1
                            posgoodpreds+=1
                            postotalpreds+=1
                        else:
                            badpreds+=1
                            totalpreds+=1
                            negbadpreds+=1
                            negtotalpreds+=1
                            
                if DataY[c][t]<0:
                    if curAlpha[c]<=0:
                        count+=1
                        
                    if curAlpha[c] != 0:
                        if curAlpha[c]<0:
                            goodpreds+=1
                            totalpreds+=1
                            neggoodpreds+=1
                            negtotalpreds+=1
                        else:
                            badpreds+=1
                            totalpreds+=1
                            posbadpreds+=1
                            postotalpreds+=1

                #1: get daily reward sum 
                curR[c]                     = np.round(curAlpha_n[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(avgDailyR[t] + curR[c], 8)

                if curAlpha_n[c]>0:
                    curR[c]  = np.round(curAlpha_n[c] * DataY[c][t], 8)
                    avgDailyR[t]                = np.round(avgDailyR[t] + curR[c], 8)
    
                    #2: pos change sum
                    posChange                   = np.round(posChange +  abs(curAlpha_n[c] - preAlpha_n[c]), 8)
                    preAlpha_n[c]               = curAlpha_n[c]
                else:
                    curR[c]  = np.round(0 * DataY[c][t], 8)
                    avgDailyR[t]                = np.round(avgDailyR[t] + curR[c], 8)
    
                    #2: pos change sum
                    posChange                   = np.round(posChange +  abs(0 - preAlpha_n[c]), 8)
                    preAlpha_n[c]               = 0
                
                
                #Feed results dataframe
                df_results.loc[c].loc[t,'Predicted Signal'] = np.sign(curAlpha[c])
                df_results.loc[c].loc[t,'Real Signal'] = np.sign(DataY[c][t])
                df_results.loc[c].loc[t,'Quantity'] = np.abs(curAlpha_n[c])
                df_results.loc[c].loc[t,'Tomorrow Returns'] = DataY[c][t]
                df_results.loc[c].loc[t,'Reward'] = curR[c]

        print('')
        print('Accuracy is {}'.format(count/((Days-1)*c)))
        print('')
        print('Number of good predictions is {} with ratio {}'.format(goodpreds,goodpreds/totalpreds))
        print('')
        print('Number of bad predictions is {} with ratio {}'.format(badpreds,badpreds/totalpreds))
        print('')
        print('Number {} and ratio of positive good predictions {}'.format(posgoodpreds, posgoodpreds/postotalpreds))
        print('')
        print('Number {} and ratio of negative good predictions {}'.format(neggoodpreds, neggoodpreds/negtotalpreds))
        print('')
        
        # calculate cumulative return
        for t in range(Days):
            cumAsset = round(cumAsset + (cumAsset * avgDailyR[t] * 0.01), 8)
            df_results.loc[c].loc[t,'Cumulative Reward at t'] = cumAsset
            print('Average daily reward is {} and cumulative asset is {}'.format(avgDailyR[t], cumAsset))
        
        print('')
        print ('cumAsset ',  cumAsset)
        
        return N, posChange, cumAsset, df_results
    
    def get_NeutralizedPortfolio(self, curA, N):         
        
        alpha       = np.zeros(N)
        avg         = 0
        
        # get average
        for c in range (N):
            alpha[c]    = 1 - np.argmax(curA[c])
            avg         = avg + alpha[c]
            
        avg     = np.round(avg / N, 4)

        #set alpha
        sum_a       = 0
        for c in range (N):
            alpha[c]= np.round(alpha[c] - avg, 4)
            sum_a   = np.round(sum_a + abs(alpha[c]), 4)

        # set alpha
        if sum_a == 0 :
            return alpha

        for c in range (N):
            alpha[c] = np.round(alpha[c] / sum_a, 8)

        return alpha

    def validate_TopBottomK_Portfolio(self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W, K):

        # list
        N           = len(DataX)
        Days        = len(DataX[0])

        # alpha
        preAlpha_s  = np.zeros(N)
        curAlpha_s  = np.zeros(N)
        posChange   = 0

        # reward
        curR        = np.zeros(N)
        avgDailyR   = np.zeros(Days)
      
        # cumulative asset: initialize curAsset to 1.0
        cumAsset = 1

        # action value for Signals and Threshold for Top/Bottom K 
        curActValue = np.zeros((N, NumAction ))
        LongSignals = np.zeros(N)

        UprTH = 0
        LwrTH = 0
        
        count = 0
        
        goodpreds = 0
        badpreds = 0
        totalpreds = 0
        
        posgoodpreds = 0
        posbadpreds = 0
        postotalpreds = 0
        
        neggoodpreds = 0
        negbadpreds = 0
        negtotalpreds = 0
        
        array_companies = []
        array_days = []
        
        for i in range(N):
            array_companies+=[i]*(Days - 1)
            array_days+=[j for j in range(Days - 1)]
        arrays = [array_companies, array_days]
        
        index = pd.MultiIndex.from_arrays(arrays, names=('Company', 'Date'))
        columns = ['Predicted Signal', 'Real Signal', 'Quantity', 'Tomorrow Returns', 'Reward', 'Cumulative Reward at t']
        df_results = pd.DataFrame(index=index, columns = columns)
        
        for t in range(Days - 1):
            for c in range(N):
           
                #1: choose action from current state 
                curS            = DataX[c][t]
                QAValues        = sess.run  (rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False })
                curActValue[c]  = np.round(QAValues[0].reshape((NumAction)), 4)
                LongSignals[c]  = curActValue[c][0] - curActValue[c][2]

            # set Top/Bottom portfolio for day t
            UprTH, LwrTH        = self.givenLongSignals_getKTH(LongSignals, K, t) 
            curAlpha_s          = self.get_TopBottomPortfolio(UprTH, LwrTH, LongSignals, N)       

            for c in range(N):
                
                if DataY[c][t]>0:
                    if curAlpha_s[c]>=0:
                        count+=1
                    if curAlpha_s[c] != 0:
                        if curAlpha_s[c]>0:
                            goodpreds+=1
                            totalpreds+=1
                            posgoodpreds+=1
                            postotalpreds+=1
                        else:
                            badpreds+=1
                            totalpreds+=1
                            negbadpreds+=1
                            negtotalpreds+=1
                            
                if DataY[c][t]<0:
                    if curAlpha_s[c]<=0:
                        count+=1
                        
                    if curAlpha_s[c] != 0:
                        if curAlpha_s[c]<0:
                            goodpreds+=1
                            totalpreds+=1
                            neggoodpreds+=1
                            negtotalpreds+=1
                        else:
                            badpreds+=1
                            totalpreds+=1
                            posbadpreds+=1
                            postotalpreds+=1

                if curAlpha_s[c]>0:
                    #1: get daily reward sum
                    curR[c]  = np.round(curAlpha_s[c] * DataY[c][t], 8)
                    avgDailyR[t] = np.round(avgDailyR[t] + curR[c], 8)
    
                    #2: pos change sum
                    posChange = np.round(posChange +  abs(curAlpha_s[c] - preAlpha_s[c]), 8)
                    preAlpha_s[c] = curAlpha_s[c]
                
                else:
                    #1: get daily reward sum
                    curR[c] = np.round(0 * DataY[c][t], 8)
                    avgDailyR[t] = np.round(avgDailyR[t] + curR[c], 8)
    
                    #2: pos change sum
                    posChange = np.round(posChange +  abs(0 - preAlpha_s[c]), 8)
                    preAlpha_s[c] = 0
                
                #Feed results dataframe
                df_results.loc[c].loc[t,'Predicted Signal'] = np.sign(curAlpha_s[c])
                df_results.loc[c].loc[t,'Real Signal'] = np.sign(DataY[c][t])
                df_results.loc[c].loc[t,'Quantity'] = np.abs(curAlpha_s[c])
                df_results.loc[c].loc[t,'Tomorrow Returns'] = DataY[c][t]
                df_results.loc[c].loc[t,'Reward'] = curR[c]
                
        print('')
        print('Number of good predictions is {} with ratio {}'.format(goodpreds,goodpreds/totalpreds))
        print('')
        print('Number of bad predictions is {} with ratio {}'.format(badpreds,badpreds/totalpreds))
        print('')
        print('Number {} and ratio of positive good predictions {}'.format(posgoodpreds, posgoodpreds/postotalpreds))
        print('')
        print('Number {} and ratio of negative good predictions {}'.format(neggoodpreds, neggoodpreds/negtotalpreds))
        print('')
        
        # calculate cumulative return
        for t in range(Days):
            cumAsset = round(cumAsset + (cumAsset * avgDailyR[t] * 0.01), 8)
            df_results.loc[c].loc[t,'Cumulative Reward at t'] = cumAsset
            print('Average daily reward is {} and cumulative asset is {}'.format(avgDailyR[t], cumAsset))
        
        print('')
        print ('cumAsset ',  cumAsset)
        
        return N, posChange, cumAsset, df_results
    
    def get_TopBottomPortfolio(self, UprTH, LwrTH, LongSignals, N):

        alpha   = np.zeros(N)
        sum_a   = 0

        for c in range (N):
            if LongSignals[c] >= UprTH:
                alpha[c] = 1
                sum_a = sum_a + 1
            elif LongSignals[c] <= LwrTH:
                alpha[c] = -1
                sum_a = sum_a+1
            else:
                alpha[c] = 0

        if sum_a == 0: 
            return alpha

        for c in range (N) :
            alpha[c] = np.round(alpha[c] / float(sum_a), 8)

        return alpha
    
    def validate_SeveralAssets_Prediction(self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W):

        # list
        N           = len(DataX)
        Days        = len(DataX[0])

        # alpha
        curAlpha  = np.zeros(N)
        
        preAlpha_v  = np.zeros(N)
        curAlpha_v  = np.zeros(N)
        
        posChange   = 0

        # reward
        curR        = np.zeros(N)
        avgDailyR   = np.zeros(Days)
      
        # cumulative asset: initialize curAsset to 1.0
        cumAsset = 1

        # action value for Signals and Threshold for Top/Bottom K 
        curA = np.zeros((N, NumAction ))
        
        count = 0
        
        goodpreds = 0
        badpreds = 0
        totalpreds = 0
        
        posgoodpreds = 0
        posbadpreds = 0
        postotalpreds = 0
        
        neggoodpreds = 0
        negbadpreds = 0
        negtotalpreds = 0
        
        array_companies = []
        array_days = []
        
        for i in range(N):
            array_companies+=[i]*(Days - 1)
            array_days+=[j for j in range(Days - 1)]
        arrays = [array_companies, array_days]
        
        index = pd.MultiIndex.from_arrays(arrays, names=('Company', 'Date'))
        columns = ['Predicted Signal', 'Real Signal', 'Quantity', 'Tomorrow Returns', 'Reward', 'Cumulative Reward at t']
        df_results = pd.DataFrame(index=index, columns = columns)
        
        for t in range (Days - 1):
            for c in range(N):
           
                #1: choose action from current state 
                curS            = DataX[c][t]
                QAValues        = sess.run  (rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False })
                curA[c]  = np.round(QAValues[0].reshape((NumAction)), 4)

            alpha = np.zeros(N)
            
            # Get Alpha (Signal -1, 0 or 1)
            for c in range (N):
                alpha[c]    = 1 - np.argmax(curA[c])
            curAlpha = alpha
            
            companies_used = 0
            
            for c in range(N):
                if curAlpha[c]>0:
                    companies_used += 1
                    curAlpha_v[c]  = curAlpha[c]
                else:
                    curAlpha_v[c] = 0
            
            for c in range(N):
                if companies_used!=0:
                    curAlpha_v[c]/=companies_used
                else:
                    curAlpha_v[c] = 0
                
            for c in range(N):

                if DataY[c][t]>=0:
                    if curAlpha[c] != 0:
                        totalpreds+=1
                        if curAlpha[c]>0:
                            goodpreds+=1
                            posgoodpreds+=1
                            postotalpreds+=1
                        else:
                            badpreds+=1
                            negbadpreds+=1
                            negtotalpreds+=1
                            
                if DataY[c][t]<=0:
                    if curAlpha[c] != 0:
                        totalpreds+=1
                        if curAlpha[c]<0:
                            goodpreds+=1
                            neggoodpreds+=1
                            negtotalpreds+=1
                        else:
                            badpreds+=1
                            posbadpreds+=1
                            postotalpreds+=1
                            
                #1: get daily reward sum
                curR[c]  = np.round(curAlpha_v[c] * DataY[c][t], 8)
                avgDailyR[t] = np.round(avgDailyR[t] + curR[c], 8)
                
                #2: pos change sum
                posChange = np.round(posChange +  abs(curAlpha_v[c] - preAlpha_v[c]), 8)
                preAlpha_v[c] = curAlpha_v[c]
                    
                
                #Feed results dataframe
                df_results.loc[c].loc[t,'Predicted Signal'] = np.sign(curAlpha[c])
                df_results.loc[c].loc[t,'Real Signal'] = np.sign(DataY[c][t])
                df_results.loc[c].loc[t,'Quantity'] = np.abs(curAlpha_v[c])
                df_results.loc[c].loc[t,'Tomorrow Returns'] = DataY[c][t]
                df_results.loc[c].loc[t,'Reward'] = curR[c]

        print('')
        print('Number of good predictions is {} with ratio {}'.format(goodpreds,goodpreds/totalpreds))
        print('')
        print('Number of bad predictions is {} with ratio {}'.format(badpreds,badpreds/totalpreds))
        print('')
        print('Number {} and ratio of positive good predictions {}'.format(posgoodpreds, posgoodpreds/postotalpreds))
        print('')
        print('Number {} and ratio of negative good predictions {}'.format(neggoodpreds, neggoodpreds/negtotalpreds))
        print('')
        # calculate cumulative return
        for t in range(Days):
            cumAsset = round(cumAsset + (cumAsset * avgDailyR[t] * 0.01), 8)
            df_results.loc[c].loc[t,'Cumulative Reward at t'] = cumAsset
            print('Average daily reward is {} and cumulative asset is {}'.format(avgDailyR[t], cumAsset))
        
        print('')
        print ('cumAsset ',  cumAsset)
        print('')
        return N, posChange, cumAsset, df_results
    
    def Test_TopBottomK_Portfolio(self, sess, saver, state, isTrain, rho_eta,  H,W, NumAction, resultsdirpath, bundle, TopK):
        
        print('')
        print('Top/Bottom K Portfolio Calculation Begins')
        
        saver.restore(sess, 'DeepQ')
        Outcome = self.validate_TopBottomK_Portfolio(self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W, TopK)
        
        print('')
        print ('NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'Total PnL: ',Outcome[2]-1)
        self.writeResult_daily(resultsdirpath, Outcome,  len (self.DataX[0])-1, bundle, method='TopBottomK')


    def Test_Neutralized_Portfolio(self, sess, saver, state, isTrain,  rho_eta,  H,W, NumAction, resultsdirpath, bundle):
        
        print('')
        print('Neutralized Portfolio Calculation Begins')
        
        saver.restore(sess, 'DeepQ')
        Outcome = self.validate_Neutralized_Portfolio(self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W)
        
        print('')
        print ('NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'Total PnL: ', Outcome[2]-1)
        self.writeResult_daily(resultsdirpath, Outcome, len(self.DataX[0])-1, bundle, method='NeutralizedPortfolio')
    
    def Test_SeveralAssets_Prediction(self, sess, saver, state, isTrain, rho_eta,  H,W, NumAction, resultsdirpath, bundle, companies_bundle):
        
        print('')
        print('Several Assets Portfolio Calculation Begins')
        
        saver.restore( sess, 'DeepQ' )
        Outcome = self.validate_SeveralAssets_Prediction(self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W)
        
        print('')
        print ('NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'Total PnL: ',Outcome[2]-1)
        self.writeResult_daily_SeveralAssets(resultsdirpath, Outcome,  len (self.DataX[0])-1, bundle, companies_bundle, method='SeveralAssetsPortfolio')


    def givenLongSignals_getKTH(self, LongSignals, K, t):
        
        Num         =  int(len(LongSignals) * K)
        SortedLongS =  np.sort(LongSignals)

        return SortedLongS[len(LongSignals) - Num], SortedLongS[Num-1]
        
    def randf(self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

    def get_randaction(self,  numofaction) :
        actvec      =  np.zeros((numofaction), dtype = np.int32)
        idx         =  random.randrange(0,numofaction)
        actvec[idx] = 1
        return actvec


    def get_reward(self, preA, curA, inputY, P):
        
        # 1,0,-1 is assined to pre_act, cur_act 
        # for action long, neutral, short respectively
        pre_act = 1- np.argmax(preA) 
        cur_act = 1- np.argmax(curA) 

        return  (cur_act * inputY) - P*abs(cur_act - pre_act) 


    def writeResult_daily(self,  resultsdirpath,  outcome, numDays, bundle, method):
        
        f = open(os.path.join(resultsdirpath,"Results"+str(method)+".txt"), 'w')
        f.write('Method: ' + method + ' and bundle: '+str(bundle)+'\n\n')
        f.write('Comp#: ' + str(outcome[0]) + ', ')
        f.write('Days#: ' + str(numDays-1) + ', ')
        f.write('TR#: ' + str(round(outcome[1]/2, 4)) + ', ')
        f.write('FinalAsset: '  + str(round(outcome[2], 4)))
        
        f.write("\n\n")
        f.close()
        
        df_results = outcome[3]
        df_results.to_csv(os.path.join(resultsdirpath,"Results_"+str(method)+"_"+str(bundle)+".csv"))
    
    def writeResult_daily_SeveralAssets(self,  resultsdirpath,  outcome, numDays, bundle, companies_bundle, method):
        
        string0, string1 = '', ''
        for company in companies_bundle:
            string0+='_'+str(company)
            string1+=str(company)+', '
        string1 = string1.strip(', ')
        
        f = open(os.path.join(resultsdirpath,"Results_"+str(method)+"_"+str(bundle)+string0+".txt"), 'w')
        
        
        f.write('Method: ' + method + ' and bundle: '+str(bundle)+ ' and companies: '+ string1 + '\n\n')
        f.write('Comp#: ' + str(outcome[0]) + ', ')
        f.write('Days#: ' + str(numDays-1) + ', ')
        f.write('TR#: ' + str(round(outcome[1]/2, 4)) + ', ')
        f.write('FinalAsset: '  + str(round(outcome[2], 4)))
        
        f.write("\n\n")
        f.close()
        
        df_results = outcome[3]
        df_results.to_csv(os.path.join(resultsdirpath,"Results_"+str(method)+"_"+str(bundle)+string0+".csv"))

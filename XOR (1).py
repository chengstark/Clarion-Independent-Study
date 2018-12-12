import os
import io
import sys
import time
import json
import logging
import random
#import Agent
#import World

class QNet(Channel):

	def __init__(self, learning_rate, momentum):
		pass
    

class QNetComponent(Component):

	def __init__(self, learning_rate, momentum):
		pass

	def initialize_knowledge(self):
		return FlowStructure(
				Flow('ACSBottom', flow_type = FlowType.BottomLevel),
				MyJunction(),
				# by passing self.param, you can allow mutations to component to be reflected in the corresponding processor
				QNet(learning_rate=self.learning_rate, momentum=self.momentum)
			)

class ACSTopLevel(Component):
	pass

class XOR:
    CorrectCounter = 0
    NumberTrials = 0
    NumberRepeats = 0
    outputfile = open("XOR.txt","w+")


    def __init__(self):
        pass

	def compute_renforcement(self, si, decision):
		if decision.chosen == Chunk("Pick True"):

            #The agent said "True".
            if ((si[Microfeature("Boolean 1", True)] == 1
                and si[Microfeature("Boolean 2", False)] == 1) or                            
				(si[Microfeature("Boolean 1", False)] == 1
                and si[Microfeature("Boolean 2", True)] == 1)):
                    
                #The agent responded correctly
                outputfile.write("PyXORJohn was correct")
                #Record the agent's success.
                self.CorrectCounter += 1
                #Give positive feedback.
                output = 1
                
            else:
                    
                #The agent responded incorrectly
                outputfile.write("PyXORJohn was incorrect")
                #Give negative feedback.
                output = 0

        else:
            
            #The agent said "False".
            if ((si[Microfeature("Boolean 1", True)] == 1
                and si[Microfeature("Boolean 2", True)] == 1) or                            
				si[(Microfeature("Boolean 1", False)] == 1
                and si[Microfeature("Boolean 2", False)] == 1)):
                    
                #The agent responded correctly
                outputfile.write("PyXORJohn was correct")
                #Record the agent's success.
                self.CorrectCounter += 1
                #Give positive feedback.
                output = 1
                    
            else:
                    
                #The agent responded incorrectly
                outputfile.write("PyXORJohn was incorrect")
                #Give negative feedback.
                output = 0

		return output
	
	
    def run():
		World.LoggingLevel = TraceLevel.Off
		PyXORJohn = Agent("PyXORJohn") #creates an empty agent
		acs1 = ACS() #creates an empty ACS
		'''
		Agent._generate_membername()
		Agent.add_subsystem()
		Agent.remvoe_subsystem()
		Agent.add_buffer()
		Agent.remove_buffer ()
		'''
		sensory_inputbuffer = SIBuffer()
		PyXORJohn.add_buffer(sensory_inputbuffer)
		selector_structure = SelectorStructure(Appraisal("Action selection"), MyJunction(), BoltzmannSelector(temperature=.01))
		    qnet_component = QNetComponent(learning_rate=1.3, momentum=.01)
		top_level_component = ACSTopLevel(rer_refinement=False)

		acs1.add_component(qnet_component) # Construct.add() calls should know how to name members 
			# class(qnet_component).__name__ == 'QNetComponent'
		acs1.add_component(top_level_component)
		acs1.network.add_selector(selector_structure) #ACS by default initializes empty Network object 
		PyXORJohn.add_acs(acs1)
		mf1 = Microfeature("Boolean 1", True)
		mf2 = Microfeature("Boolean 1", False)
		mf3 = Microfeature("Boolean 2", True)
		mf4 = Microfeature("Boolean 2", False)

		ch1 = Chunk("Pick True")
		ch2 = Chunk("Pick False")

        double r = null
        si = ActivationPacket()
        for j in range(0, NumberRepeats):
            for i in range(0, NumberTrials):
                outputfile.write("Running trial #", (i+1), " of block #"+ (j+1))
                r = random.uniform(0, 1)

                if (r < 0.25):
                    #True:True
					si[mf1] = 1
					si[mf2] = 0
					si[mf3] = 1
					si[mf4] = 0
                
                else if (r < 0.5):
                    #True:False
					si[mf1] = 1
					si[mf2] = 0
					si[mf3] = 0
					si[mf4] = 1

                else if (r < 0.75):
                    #False:True
					si[mf1] = 0
					si[mf2] = 1
					si[mf3] = 1
					si[mf4] = 0

                else:
                    #False:False
                    si[mf1] = 0
					si[mf2] = 1
					si[mf3] = 0
					si[mf4] = 1

                PyXORJohn.set_si(si)
				PyXORJohn.cycle()
                chosen = PyXORJohn.subsystems["acs"].network.get_appraisal()

                
                if chosen:    
					PyXORJohn.learn()
            
            ReportResults(j+1)

            self.CorrectCounter = 0
        
        outputfile.close()
        print("PyXORJohn has completed the task", repeatCount)
        print("Killing PyXORJohn")
        #PyXORJohn.Die()
        #print("PyXORJohn is Dead")
        print("XOR Task Completed. See XOR.txt for Results")
        input("Press any key to exit")


        def ReportResults(repeatCount):
            pass


                        
                    


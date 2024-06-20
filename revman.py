
''' In class game to motivate Revenue Management and Littlewood's rule/Newsvendor problem
Adapted from a pen-and-paper version by Jim Schummer

'''
# from  tkinter import *
# from  tkinter import ttk
import tkinter as tk
import pandas as pd

import numpy as np
from scipy.interpolate import interp1d
# import scipy.opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Demand:
    def __init__(self, Pmax = 10, Qmax = 30, steps = 7, capacity = 5, seed = None):
        self.Pmax = Pmax
        self.Qmax = Qmax
        self.steps = steps
        self.capacity = capacity
        self.rand = np.random.default_rng(seed)
        self.price_points = None
        self.hidden = True
        self.makefig()



    def makefig(self):
        self.fig, self.ax = plt.subplots(dpi = 100)
        self.ax.set_xlim(0, self.steps + 2)
        self.hidden = False

    def add_points(self, points):
        self.price_points = np.array(points)
        self.ax.set_ylim(0, np.max(self.price_points)+1)

    def draw_new(self, mult = 1):
        try:
            self.current_demand = np.sort(self.rand.choice(
                                            mult*self.price_points,
                                            size=self.steps,
                                            replace=False) )
            if mult > 1:
                self.ax.set_ylim(0, np.max(self.current_demand)+1)

            print(self.current_demand)
        except ValueError:
            print('price points not set')



        
    def evaluate_price(self, choices):
        '''value error is caught upstream?'''
        self.choices = np.array(choices)
        self.choices = np.round(choices,2)
        quantities = len(self.current_demand) - np.searchsorted(self.current_demand, self.choices)
        profits = self.choices*quantities
        self.scores = np.round(profits,3)
        return self.scores

    def evaluate_capacity(self, choices, p1, p2, d1, d2):
        self.choices = np.array(choices,dtype = int)
        if np.min(self.choices) < 0 or np.max(self.choices) > self.capacity:
            raise ValueError
        max_sales = len(d1) - np.searchsorted(d1, p1)
        max_sales2 = len(d2) - np.searchsorted(d2, p2)
        sales1 = np.minimum(self.capacity - self.choices, max_sales)
        sales2 = np.minimum(max_sales2, np.maximum(self.choices, self.capacity - sales1))
        print(f'Market A sales: {sales1}, Market B sales: {sales2}')
        self.scores = sales1*p1 + sales2*p2
        return self.scores




        
    def plot(self):
        ''' expects initialized ax and fig '''
        demand = np.append(self.current_demand[::-1],0)
        demand = np.append(self.current_demand[-1],demand)
        colors = plt.cm.rainbow(np.linspace(0,1, len(self.choices)))
        if not hasattr(self, 'lines_dict'):
            demand_l, = self.ax.step(np.arange(0,self.steps+2),demand, label = 'demand', color = 'black')
            self.lines_dict = dict(demand_l=demand_l)
            for i,c in enumerate(self.choices):
                self.lines_dict[f'player_{i+1}_price'] = self.ax.axhline(c, label = f'player {i+1}', color = colors[i], linestyle = ':')

            self.ax.legend()
        else:
            self.lines_dict['demand_l'].set_data(np.arange(0,self.steps+2), demand)
            for i,c in enumerate(self.choices):
                self.lines_dict[f'player_{i+1}_price'].set_ydata([c, c])

    def toggle_hide(self):
        [l.set_visible(not l.get_visible()) for k,l in self.lines_dict.items()]

        self.hidden = not self.hidden
        # return fig, ax

        
class Game:
    def __init__(self, rounds = 4, players = 4, ticks = 10, test = False, demandkwargs = None):
        '''constants'''
        self.max_rounds = rounds
        self.n_players = players
        self.n_ticks = ticks
        '''state'''
        self.current_round = 0
        # TODO multi
        self.running = np.zeros(self.n_players)
        self.fresh_draw = False
        self.choice_data = pd.DataFrame(index = range(1,self.n_players+1))

        ''' game parameters '''
        if demandkwargs: self.game = Demand(**demandkwargs)
        else: self.game = Demand()

        ''' plot parameters '''

        ''' grid'''


        self.root = tk.Tk()
        self.root.title('Volatile Times')
        self.frame = tk.Frame(self.root)
        self.frame.grid()

        '''game state'''

        DISPLAY_ROWS = 0
        DISPLAY_COLUMNS = 0

        self.ticks = [tk.StringVar() for _ in range(self.n_ticks)]

        tk.Label(master = self.frame, text = f"Price points").grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS)
        for i in range(self.n_ticks):
            tk.Entry(master = self.frame, width = 4, textvariable = self.ticks[i]).grid(column=DISPLAY_COLUMNS +0, row=DISPLAY_ROWS+i+1)
        tk.Button(master = self.frame, text = 'Set points', command = self.set_points).grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS+ i +2, rowspan = self.n_players)

        
        DISPLAY_COLUMNS+=2

        tk.Label(master = self.frame, text = f"Demand").grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS)
        self.demand_pts = [tk.StringVar() for _ in range(self.game.steps)]
        for i in range(self.game.steps):
            tk.Label(master = self.frame, width = 4, textvariable = self.demand_pts[i]).grid(column=DISPLAY_COLUMNS +0, row=DISPLAY_ROWS+i+1)
        # tk.Button(master = self.frame, text = 'Set points', command = self.set_points).grid(column = DISPLAY_COLUMNS + 2, row = DISPLAY_ROWS, rowspan = self.n_players)



        tk.Label(master = self.frame, text = '').grid(column = DISPLAY_COLUMNS + 1, row = DISPLAY_ROWS+2, pady =3) # blank space

        DISPLAY_ROWS+=3
        DISPLAY_COLUMNS+=4

        '''interface'''


        # TODO multi. make list of choices
        self.choices = [tk.StringVar() for _ in range(self.n_players)]


        tk.Button(master = self.frame, text = 'Play!', command = self.score_calc).grid(column = DISPLAY_COLUMNS + 2, row = DISPLAY_ROWS, rowspan = self.n_players)
        for i in range(self.n_players):
            tk.Label(master = self.frame, text = f"Team {i+1} Price").grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS)
            tk.Entry(master = self.frame, width = 4, textvariable = self.choices[i]).grid(column=DISPLAY_COLUMNS +1, row=DISPLAY_ROWS)
            DISPLAY_ROWS += 1


        
        # isa_entry = (tk.Entry(master = self.frame, 
                             # width = 4, 
                             # textvariable = self.isa_choice)
                       # .grid(column=1, row=DISPLAY_ROWS)
                     # )
        # player_entry = (tk.Entry(master = self.frame, 
                                # width = 4, 
                                # textvariable = self.player_choice)
                          # .grid(column=1, row=DISPLAY_ROWS+1)
                          # )




        '''Variables'''


        DISPLAY_ROWS+=2 

        tk.Label(master = self.frame, text = 'SCORECARD').grid(column = DISPLAY_COLUMNS +0, row = DISPLAY_ROWS, columnspan = 3, padx = 3, pady = 40)
        self.scores = [[tk.StringVar() for r in range(self.max_rounds)] for _ in range(self.n_players)]  # player as row, 1st index

        # self.isa_scores = [tk.StringVar() for r in range(self.max_rounds)]
        # self.player_scores = [tk.StringVar() for r in range(self.max_rounds)]
        self.totals = [tk.StringVar() for _ in range(self.n_players)]
        # self.total_isa, self.total_player = tk.StringVar(), tk.StringVar()
        self.winner = tk.StringVar()

        DISPLAY_ROWS+=1

        '''Score Keeping'''

        tk.Label(master = self.frame, text = f'TOTALS').grid(column = DISPLAY_COLUMNS + self.max_rounds + 1, row = DISPLAY_ROWS, padx = 6, pady = 5)
        for p in range(self.n_players): # total numbers
            tk.Label(master = self.frame, textvariable = self.totals[p], relief = 'sunken', width =6).grid(column = DISPLAY_COLUMNS + self.max_rounds+1, row = DISPLAY_ROWS + p + 1, padx = 6)
        for r in range(self.max_rounds): # round labels
            tk.Label(master = self.frame, text = f'Round {r + 1}', width = 7).grid(row = DISPLAY_ROWS, column = DISPLAY_COLUMNS + 1 + r, padx = 3)

        # roundwise numbers
        DISPLAY_ROWS+=1
        for p in range(self.n_players):
            # team labels
            tk.Label(master = self.frame, text = f'Team {p+1}').grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS + p, padx = 3)
            for r in range(self.max_rounds): # additional is for label
                tk.Label(master = self.frame, textvariable = self.scores[p][r], relief = 'sunken', width = 6).grid(row = DISPLAY_ROWS + p, column = DISPLAY_COLUMNS + 1 + r, padx = 3)
        # tk.Label(master = self.frame, textvariable = self.total_player, relief = 'sunken', width =6).grid(column = DISPLAY_COLUMNS + 2, row = DISPLAY_ROWS, padx = 3)

        DISPLAY_ROWS += self.n_players + 2

        tk.Label(master = self.frame, textvariable = self.winner).grid(column = DISPLAY_COLUMNS + 2, row = DISPLAY_ROWS, columnspan = 3, padx = 3, pady = 5)



        '''Choice Keeping'''

        DISPLAY_ROWS += 1
        tk.Label(master = self.frame, text = 'PAST PRICES').grid(column = DISPLAY_COLUMNS +2, row = DISPLAY_ROWS, columnspan = 3, padx = 3, pady = 8)
        DISPLAY_ROWS += 1
        self.past_prices = [[tk.StringVar() for r in range(self.max_rounds)] for _ in range(self.n_players)]  # player as row, 1st index
        for r in range(self.max_rounds): # round labels
            tk.Label(master = self.frame, text = f'Round {r + 1}', width = 7).grid(row = DISPLAY_ROWS, column = DISPLAY_COLUMNS + 1 + r, padx = 3)

        # roundwise numbers
        DISPLAY_ROWS+=1
        for p in range(self.n_players):
            # team labels
            tk.Label(master = self.frame, text = f'Team {p+1}').grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS + p, padx = 3)
            for r in range(self.max_rounds): # additional is for label
                tk.Label(master = self.frame, textvariable = self.past_prices[p][r], relief = 'sunken', width = 6).grid(row = DISPLAY_ROWS + p, column = DISPLAY_COLUMNS + 1 + r, padx = 3)
        # tk.Label(master = self.frame, textvariable = self.total_player, relief = 'sunken', width =6).grid(column = DISPLAY_COLUMNS + 2, row = DISPLAY_ROWS, padx = 3)

        DISPLAY_ROWS += self.n_players + 2


        '''new game'''

        DISPLAY_ROWS+=1

        tk.Button(master = self.frame, text = 'New Game', command = self.restart).grid(column = DISPLAY_COLUMNS + 0, row = DISPLAY_ROWS, columnspan= 3, padx = 3, pady = 10)


        ''' display '''

        self.canvas = FigureCanvasTkAgg(self.game.fig, master = self.frame)
        self.canvas.get_tk_widget().grid(column = DISPLAY_COLUMNS + self.max_rounds + 3, row = 0, rowspan = DISPLAY_ROWS)
        pass



    def play(self):
        ''' define vars '''
        self.root.mainloop()
        ### implement rounds later, subtle, needs own event loop (or not)?

    def draw_new(self):
        '''waits for play before redrawing--do i want this default???'''
        if self.fresh_draw: return
        ''' clear choices '''
        '''capture somehow into csv'''
        [c.set('') for c in self.choices]
        '''announce params'''
        self.game.draw_new()
        # if self.current_round > 0 and not self.game.hidden:
            # self.game.toggle_hide()
            # self.canvas.draw()
        self.fresh_draw = True

    def set_points(self):
        self.price_points = [float(c.get()) for c in self.ticks]
        self.game.add_points(self.price_points)
        self.canvas.draw()


    def score_calc(self):
        try:
            ## TODO: enforced rounding
            if self.current_round <= self.max_rounds-2:
                '''choices'''
                '''capture somehow into csv'''
                # TODO forloop over choices
                choices = [float(c.get()) for c in self.choices]
                if self.current_round == self.max_rounds -2:
                    mult = 2
                else: mult = 1
                print(f'current round is {self.current_round}, mult is {mult}')

                self.game.draw_new(mult)
                scores = self.game.evaluate_price(choices)


                dats = {f'choice_{self.current_round+1}': choices}
                        # , f'score_{self.current_round+1}': scores}
                self.choice_data = (self.choice_data.assign(**dats))
                print(self.choice_data.head())

                self.game.plot()# carries state, poor design?
                for p in range(self.n_players):
                    self.scores[p][self.current_round].set(scores[p])
                    self.past_prices[p][self.current_round].set(choices[p])
                if self.game.hidden: self.game.toggle_hide()
                self.canvas.draw()

                '''make this part numpy'''
                # self.running *= (self.current_round)/(self.current_round + 1)
                self.running += scores
                # /(self.current_round+1)
                rounded = np.round(self.running, 3)
                [t.set(rounded[i]) for i,t in enumerate(self.totals)]
                self.current_round += 1
                self.fresh_draw = False

            elif self.current_round == self.max_rounds-1:
                choices = [float(c.get()) for c in self.choices]
                self.game.draw_new(1)
                d1 = self.game.current_demand
                self.game.draw_new(2)
                d2 = self.game.current_demand
                p1 = self.choice_data.iloc[:,-2].values
                p2 = self.choice_data.iloc[:,-1].values
                print(f'd1:{d1},d2:{d2},p1:{p1},p2:{p2}')

                scores = 2*self.game.evaluate_capacity(choices, p1, p2, d1, d2)

                self.game.plot()# carries state, poor design?
                for p in range(self.n_players):
                    self.scores[p][self.current_round].set(scores[p])
                if self.game.hidden: self.game.toggle_hide()
                self.canvas.draw()
                self.running += scores
                rounded = np.round(self.running, 3)
                [t.set(rounded[i]) for i,t in enumerate(self.totals)]
                self.current_round += 1


                best = np.max(self.running)
                winners = [i+1 for i in range(self.n_players) if self.running[i] == best]
                if len(winners) > 1:
                    self.winner.set(f'Teams {", ".join(str(i) for i in winners)} tie for the win')
                else:
                    win_unique, =  winners
                    self.winner.set(f'Team {win_unique} wins!')

        except ValueError as e:
            print(e)
            print('Invalid input, try again')

        pass

    # this is old code
    def _score_calc(self):
        '''Interesting, no 'while' needed to keep 'asking' for input while wrong.
            this all gets taken care of by event loop: nothing happens when exception thrown'''
        try:
            ## TODO: enforced rounding
            if self.current_round < self.max_rounds and self.fresh_draw:
                '''choices'''
                '''capture somehow into csv'''
                # TODO forloop over choices
                choices = [float(c.get()) for c in self.choices]
                scores = self.game.evaluate(choices)
                self.game.plot()# carries state, poor design?
                for p in range(self.n_players):
                    self.scores[p][self.current_round].set(scores[p])
                if self.game.hidden: self.game.toggle_hide()
                self.canvas.draw()

                '''make this part numpy'''
                self.running *= (self.current_round)/(self.current_round + 1)
                self.running += scores/(self.current_round+1)
                rounded = np.round(self.running, 3)
                [t.set(rounded[i]) for i,t in enumerate(self.totals)]
                self.current_round += 1
                self.fresh_draw = False
            elif self.current_round == self.max_rounds:
                best = np.max(self.running)
                winners = [i+1 for i in range(self.n_players) if self.running[i] == best]
                if len(winners) > 1:
                    self.winner.set(f'Teams {", ".join(str(i) for i in winners)} tie for the win')
                else:
                    win_unique, =  winners
                    self.winner.set(f'Team {win_unique} wins!')

        except ValueError:
            print('Invalid input, try again')



    def restart(self):
        [c.set('') for c in self.choices]
        [ro.set('') for pl in self.scores for ro in pl]
        [pts.set('') for pts in self.ticks]
        [ro.set('') for pl in self.past_prices for ro in pl]
        self.current_round = 0
        self.game.toggle_hide()
        self.canvas.draw()
        self.fresh_draw = False
        self.winner.set('')
        self.running = np.zeros(self.n_players)






def test_stochast(n = 2000):
    dem = Demand()
    for _ in range(n):
        dem.draw_new()
        i_dums = dem.cost + (dem.p_choke - dem.cost)*dem.rand.random()
        p_dums = dem.cost + (dem.p_choke - dem.cost)*dem.rand.random()
        dem.evaluate(i_dums, p_dums)

# def calc_stats(
if __name__ == "__main__":
    print('Number of teams? ')
    players = int(input())
    print('Number of price points? ')
    ticks = int(input())
    print('Number of steps? ')
    steps = int(input())
    # dem = Demand()
    # dem.draw_new()
    # dem.evaluate(np.linspace(dem.cost+1e-3,dem.p_choke,5))
    # dem.plot()
    # plt.show()
    a = Game(players = players, ticks = ticks, demandkwargs = {'steps':steps})
    # a = Game(players = 2, ticks = 5, demandkwargs = {'steps':3})
    a.play()


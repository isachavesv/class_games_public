''' 

Class game introducing robust low-information pricing rule from

''A simple rule for pricing with limited knowledge of demand''
MC Cohen, G Perakis, RS Pindyck - Management Science, 2021.

The authors show good robustness properties of using the rule price = (choke price + marginal cost)/2
for situations where demand is unknown, capacity is unconstrained, but marginal costs and choke prices can be estimated.

Students typically intuit the rule but overfit to minor random variations. By the end, most have deviated from the rule.

'''


# from  tkinter import *
# from  tkinter import ttk
import tkinter as tk

import numpy as np
from scipy.interpolate import interp1d
# import scipy.opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Demand:
    def __init__(self, Pmax = 10, Qmax = 30, Cmax = 6, grid_n = 5, hints = False, seed = None):
        self.Pmax = Pmax
        self.Qmax = Qmax
        self.Cmax = Cmax
        self.grid_n = grid_n
        self.rand = np.random.default_rng(seed)
        self._makefig()
        self.hints = hints


    def _makefig(self):
        self.fig, self.ax = plt.subplots(dpi = 100)
        self.ax.set_xlim(0, self.Qmax)
        self.ax.set_ylim(0, self.Pmax)
        self.hidden = False

    def draw_new(self):
        '''choice prices only in cents'''
        '''perhaps draw cost and choke separately?'''
        ''' maybe constrain choke and cost to be 1/4 dollars'''
        q_draws = np.sort(self.Qmax*self.rand.random(self.grid_n))
        p_draws = np.sort(self.Pmax*self.rand.random(self.grid_n))
        p_draws[-1] = np.round(p_draws[-1],2)
        q_pts = np.append(0,q_draws)
        p_pts = np.append(0, p_draws)
        self.p_choke = p_pts[-1]
        self.cost = np.round(p_pts[1]/2 + (p_pts[2] - p_pts[1]/2)*self.rand.random(),2)


        self.demand = interp1d(p_pts, q_pts[::-1])
        # rounding below NOT redundant. np.arange does not "round" for even spaces due to floating pt ops
        self.price_tries = np.round(np.arange(self.cost, self.p_choke,0.01),2) 
        self.profits = (self.price_tries - self.cost)*self.demand(self.price_tries)
        opt_index = np.argmax(self.profits)
        self.best_price = self.price_tries[opt_index]
        self.best_profit = self.profits[opt_index]
        if self.hints:
            print(f'best case profits are is {self.best_profit}',  
                  f'choke is {self.p_choke}',
                  f'cost is {self.cost}',
                  f'mid is {(self.p_choke + self.cost)/2}')

        
    def evaluate(self, choices):
        # TODO: multi
        '''value error is caught upstream?'''
        self.choices = np.array(choices)
        self.choices = np.round(choices,2)
        if min(self.choices) < self.cost: raise ValueError
        profits = (self.choices - self.cost)*self.demand(self.choices)
        self.scores = np.round(profits/self.best_profit,3)
        return self.scores



        
    def plot(self):
        ''' expects initialized ax and fig '''
        prices = np.linspace(0, self.p_choke, 300)
        demands = self.demand(prices)
        colors = plt.cm.rainbow(np.linspace(0,1, len(self.choices)))
        print(colors)
        if not hasattr(self, 'lines_dict'):
            cost_l = self.ax.axhline(self.cost, label = 'marg. cost', color = 'black', linestyle = '--')
            demand_l, = self.ax.plot(demands, prices, label = 'demand', color = 'black')
            opt_price_l = self.ax.axhline(self.best_price, label = 'optimal price', color = 'green', linestyle = '--')

            self.lines_dict = dict(
                                    demand_l=demand_l,
                                    opt_price_l = opt_price_l,
                                    cost_l=cost_l)
            for i,c in enumerate(self.choices):
                self.lines_dict[f'player_{i}_price'] = self.ax.axhline(c, label = f'player {i}', color = colors[i], linestyle = ':')

            self.ax.legend()
        else:
            self.lines_dict['demand_l'].set_data(demands, prices)
            self.lines_dict['opt_price_l'].set_ydata([self.best_price, self.best_price])
            self.lines_dict['cost_l'].set_ydata([self.cost, self.cost])
            for i,c in enumerate(self.choices):
                self.lines_dict[f'player_{i}_price'].set_ydata([c, c])

    def toggle_hide(self):
        [l.set_visible(not l.get_visible()) for k,l in self.lines_dict.items()]
        self.hidden = not self.hidden
        # return fig, ax

        
class Game:
    def __init__(self, rounds = 4, players = 4, demandkwargs = None):
        '''constants'''
        self.max_rounds = rounds
        self.n_players = players
        '''state'''
        self.current_round = 0
        # TODO multi
        self.running = np.zeros(self.n_players)
        self.fresh_draw = False

        ''' game parameters '''
        if demandkwargs: self.game = Demand(**demandkwargs)
        else: self.game = Demand()

        ''' plot parameters '''

        ''' grid'''


        self.root = tk.Tk()
        self.root.title('Pricing, Sight Unseen')
        self.frame = tk.Frame(self.root)
        self.frame.grid()

        '''game state'''

        DISPLAY_ROWS = 0

        self.today_choke, self.today_cost = tk.StringVar(), tk.StringVar()

        tk.Label(master = self.frame, text = "Choke Price").grid(column = 0, row = DISPLAY_ROWS)
        tk.Label(master = self.frame, text = "Marginal Unit Cost").grid(column = 0, row = DISPLAY_ROWS+1)


        tk.Label(master = self.frame, textvariable = self.today_choke, relief = 'sunken', width = 5).grid(column = 1, row = DISPLAY_ROWS, padx = 3)
        tk.Label(master = self.frame, textvariable = self.today_cost, relief = 'sunken', width = 5).grid(column = 1, row = DISPLAY_ROWS +1, padx = 3)
        tk.Button(master = self.frame, text = 'Hints', command = self.draw_new).grid(column = 2, row = DISPLAY_ROWS, rowspan = 2)

        tk.Label(master = self.frame, text = '').grid(column = 1, row = DISPLAY_ROWS+2, pady =3) # blank space

        DISPLAY_ROWS+=3

        '''interface'''


        # TODO multi. make list of choices
        self.isa_choice , self.player_choice = tk.StringVar(), tk.StringVar()
        self.choices = [tk.StringVar() for _ in range(self.n_players)]


        tk.Button(master = self.frame, text = 'Play!', command = self.score_calc).grid(column = 2, row = DISPLAY_ROWS, rowspan = self.n_players)
        for i in range(self.n_players):
            tk.Label(master = self.frame, text = f"Team {i+1} Price").grid(column = 0, row = DISPLAY_ROWS)
            tk.Entry(master = self.frame, width = 4, textvariable = self.choices[i]).grid(column=1, row=DISPLAY_ROWS)
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

        tk.Label(master = self.frame, text = 'SCORECARD').grid(column =0, row = DISPLAY_ROWS, columnspan = 3, padx = 3, pady = 40)
        self.scores = [[tk.StringVar() for r in range(self.max_rounds)] for _ in range(self.n_players)]  # player as row, 1st index

        # self.isa_scores = [tk.StringVar() for r in range(self.max_rounds)]
        # self.player_scores = [tk.StringVar() for r in range(self.max_rounds)]
        self.totals = [tk.StringVar() for _ in range(self.n_players)]
        # self.total_isa, self.total_player = tk.StringVar(), tk.StringVar()
        self.winner = tk.StringVar()

        DISPLAY_ROWS+=1

        '''Score Keeping'''

        tk.Label(master = self.frame, text = f'TOTALS').grid(column = self.max_rounds + 1, row = DISPLAY_ROWS, padx = 6, pady = 5)
        for p in range(self.n_players): # total numbers
            tk.Label(master = self.frame, textvariable = self.totals[p], relief = 'sunken', width =6).grid(column = self.max_rounds+1, row = DISPLAY_ROWS + p + 1, padx = 6)
        for r in range(self.max_rounds): # round labels
            tk.Label(master = self.frame, text = f'Round {r + 1}', width = 6).grid(row = DISPLAY_ROWS, column = 1 + r, padx = 3)

        # roundwise numbers
        DISPLAY_ROWS+=1
        for p in range(self.n_players):
            # team labels
            tk.Label(master = self.frame, text = f'Team {p+1}').grid(column = 0, row = DISPLAY_ROWS + p, padx = 3)
            for r in range(self.max_rounds): # additional is for label
                tk.Label(master = self.frame, textvariable = self.scores[p][r], relief = 'sunken', width = 6).grid(row = DISPLAY_ROWS + p, column = 1 + r, padx = 3)
        # tk.Label(master = self.frame, textvariable = self.total_player, relief = 'sunken', width =6).grid(column = 2, row = DISPLAY_ROWS, padx = 3)

        DISPLAY_ROWS += self.n_players + 2

        tk.Label(master = self.frame, textvariable = self.winner).grid(column = 0, row = DISPLAY_ROWS, columnspan = 3, padx = 3, pady = 5)

        DISPLAY_ROWS+=1

        tk.Button(master = self.frame, text = 'New Game', command = self.restart).grid(column = 0, row = DISPLAY_ROWS, columnspan= 3, padx = 3, pady = 10)


        ''' display '''

        self.canvas = FigureCanvasTkAgg(self.game.fig, master = self.frame)
        self.canvas.get_tk_widget().grid(column = self.max_rounds + 3, row = 0, rowspan = DISPLAY_ROWS)
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
        self.today_choke.set(self.game.p_choke)
        self.today_cost.set(self.game.cost)
        if self.current_round > 0 and not self.game.hidden:
            self.game.toggle_hide()
            self.canvas.draw()
        self.fresh_draw = True

    def score_calc(self):
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
        [t.set('') for t in self.totals]
        self.current_round = 0
        self.game.toggle_hide()
        self.canvas.draw()
        self.fresh_draw = False
        self.today_cost.set('')
        self.today_choke.set('')
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
    pass
    print('Number of teams? ')
    players = int(input())
    print('Number of rounds? ')
    rounds = int(input())
    print('Number of grid points? ')
    grid_n = int(input())
    print('Show double-secret hints?')
    hints = int(input())
    # dem = Demand()
    # dem.draw_new()
    # dem.evaluate(np.linspace(dem.cost+1e-3,dem.p_choke,5))
    # dem.plot()
    # plt.show()
    a = Game(players = players, rounds = rounds, demandkwargs = {'grid_n': grid_n, 'hints': hints})
    a.play()


:- use_module('metagol').
:- style_check(-singleton).

width(2).
height(2).

metagol:min_clauses(1).
metagol:max_clauses(4).

metarule([P,Q], [P,A,B], [[Q,B,A]]).
metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]).
metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).

body_pred(up/2).
body_pred(down/2).
body_pred(left/2).
body_pred(right/2).
body_pred(bomb/1).
body_pred(sea/1).
body_pred(chess/1).
body_pred(lowest/2).
body_pred(far/2).
body_pred(jump/2).
body_pred(terminate/2).

up([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-
    height(H),
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    Y0 < H,
    Y1 is Y0 + 1.

down([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    Y0 > 0,
    Y1 is Y0 - 1.

left([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    X0 > 0,
    X1 is X0 - 1.

right([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-
    width(W),
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    X0 < W,
    X1 is X0 + 1.

jump([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    abs(X1-X0) =:= 1,
    abs(Y1-Y0) =:= 1.

terminate([[X0,Y0]], [[X1,Y1]|T]) :-
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    X1 is X0,
    Y1 is Y0.

lowest([[X0,Y0]], B) :-
    number(Y0),
    Y0 is 0.

sea([A,B,C]) :-
    number(C),
    C is 3.

chess([A,B,C]) :-
    number(C),
    C > 3.

bomb([A,B,C]) :-
    number(C),
    B is 1.

far([[X0,Y0],[X1,Y1]|_], [[X2,Y2]|T]) :-
    number(X0),
    number(X1),
    number(Y0),
    number(Y1),
    number(X2),
    number(Y2),
    abs(X2-X1)+abs(Y2-Y1) > abs(X2-X0)+abs(Y2-Y0).


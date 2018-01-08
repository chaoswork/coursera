%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg = MESSAGES;
if isMax
  for i=1:N
    P.cliqueList(i).val = log(P.cliqueList(i).val);
  endfor
  FactorFunc = @FactorSum;
  FactorMarginalFunc = @FactorMaxMarginalization;
else
  FactorFunc = @FactorProduct;
  FactorMarginalFunc = @FactorMarginalization;
endif

for t=1:2*(N-1)
  [i, j] = GetNextCliques(P, msg);
  %printf("i=%d,j=%d\n", i, j);
  msg_to_i = struct('var', [], 'card', [], 'val', []);
  for k=1:N
    if P.edges(i, k) && k!=j
	msg_to_i = FactorFunc(msg_to_i, msg(k, i));
    endif
  endfor
  %disp(msg(i, j));
  msg(i, j) = FactorFunc(P.cliqueList(i), msg_to_i);
  reduVars = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
  msg(i, j) =  FactorMarginalFunc(msg(i, j), reduVars);
  if isMax == 0
    msg(i, j).val = msg(i, j).val / sum(msg(i, j).val);
  endif
  %disp(msg(i, j));
endfor

for i=1:N
  for j=1:N
    if P.edges(i, j)
      P.cliqueList(i) = FactorFunc(P.cliqueList(i), msg(j, i));
    endif
  endfor
endfor

return

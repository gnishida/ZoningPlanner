M = csvread('features.txt');
[pc,score,latent,tsquare] = princomp(M);
%pc
%cumsum(latent)./sum(latent)
%biplot(pc(:,1:2),'Scores',score(:,1:2),'VarLabels',{'Dx' 'Dy' 'Dt' 'Sx' 'Sy' 'St' 'Bx' 'By' 'Bt' 'Tx' 'Ty' 'Tt' 'L1x' 'L1y' 'L1t' 'L2x' 'L2y' 'L2y' '?'})
%biplot(pc(:,1:2),'Scores',score(:,1:2),'VarLabels',{'Dx' 'Dy' 'Dt' 'S1x' 'S1y' 'S1t' 'S2x' 'S2y' 'S2t' 'B1x' 'B1y' 'B1t' 'B2x' 'B2y' 'B2t' 'B3x' 'B3y' 'B3t' 'Tx' 'Ty' 'Tt' 'L1x' 'L1y' 'L1t' 'L2x' 'L2y' 'L2y' 'L3x' 'L3y' 'L3y' 'L4x' 'L4y' 'L4y' 'Px' 'Py' 'Pt' 'Ex' 'Ey' 'Et' '?'})
biplot(pc(:,1:2),'Scores',scores(:,1:2),'VarLabels',{'St' 'Sc' 'Re' 'Pa' 'Am' 'Li' 'No' 'Po' 'Tr'})

%figure()
%plot(score(:,1),score(:,2),'+')
%xlable('1st principal component')
%ylable('2nd principal component')
    
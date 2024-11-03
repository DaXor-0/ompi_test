#include "coll_base_swing_utils.h"
#include <stdlib.h>

const int send_2[2][1] = {{1},{0}};
const int recv_2[2][1] = {{0},{1}};

const int send_4[4][2] = {{2,1},{0,3},{0,2},{2,0}};
const int recv_4[4][2] = {{0,0},{2,2},{2,3},{0,1}};

const int send_8[8][3] = {{4,2,1},{0,6,5},{0,4,7},{4,2,0},{4,0,3},{0,4,6},{0,6,4},{4,0,2}};
const int recv_8[8][3] = {{0,0,0},{4,4,4},{4,6,6},{0,0,1},{0,2,2},{4,6,7},{4,4,5},{0,2,3}};

const int send_16[16][4] = {{8,4,2,1},{0,12,10,9},{0,8,14,13},{8,4,0,3},{8,0,6,5},{0,8,12,15},{0,12,10,8},{8,0,4,7},{8,4,0,2},{0,12,8,11},{0,8,12,14},{8,4,2,0},{8,0,4,6},{0,8,14,12},{0,12,8,10},{8,0,6,4}};
const int recv_16[16][4] = {{0,0,0,0},{8,8,8,8},{8,12,12,12},{0,0,2,2},{0,4,4,4},{8,12,14,14},{8,8,8,9},{0,4,6,6},{0,0,2,3},{8,8,10,10},{8,12,14,15},{0,0,0,1},{0,4,6,7},{8,12,12,13},{8,8,10,11},{0,4,4,5}};

const int send_32[32][5] = {{16,8,4,2,1},{0,24,20,18,17},{0,16,28,26,25},{16,8,0,6,5},{16,0,12,8,11},{0,16,24,30,29},{0,24,20,16,19},{16,0,8,14,13},{16,8,0,4,7},{0,24,16,22,21},{0,16,24,28,31},{16,8,4,2,0},{16,0,8,12,15},{0,16,28,26,24},{0,24,16,20,23},{16,0,12,8,10},{16,8,4,0,3},{0,24,20,16,18},{0,16,28,24,27},{16,8,0,4,6},{16,0,12,10,9},{0,16,24,28,30},{0,24,20,18,16},{16,0,8,12,14},{16,8,0,6,4},{0,24,16,20,22},{0,16,24,30,28},{16,8,4,0,2},{16,0,8,14,12},{0,16,28,24,26},{0,24,16,22,20},{16,0,12,10,8}};
const int recv_32[32][5] = {{0,0,0,0,0},{16,16,16,16,16},{16,24,24,24,24},{0,0,4,4,4},{0,8,8,10,10},{16,24,28,28,28},{16,16,16,18,18},{0,8,12,12,12},{0,0,4,6,6},{16,16,20,20,20},{16,24,28,30,30},{0,0,0,0,1},{0,8,12,14,14},{16,24,24,24,25},{16,16,20,22,22},{0,8,8,10,11},{0,0,0,2,2},{16,16,16,18,19},{16,24,24,26,26},{0,0,4,6,7},{0,8,8,8,8},{16,24,28,30,31},{16,16,16,16,17},{0,8,12,14,15},{0,0,4,4,5},{16,16,20,22,23},{16,24,28,28,29},{0,0,0,2,3},{0,8,12,12,13},{16,24,24,26,27},{16,16,20,20,21},{0,8,8,8,9}};

const int send_64[64][6] = {{32,16,8,4,2,1},{0,48,40,36,34,33},{0,32,56,52,50,49},{32,16,0,12,10,9},{32,0,24,16,22,21},{0,32,48,60,58,57},{0,48,40,32,38,37},{32,0,16,28,24,27},{32,16,0,8,14,13},{0,48,32,44,40,43},{0,32,48,56,62,61},{32,16,8,4,0,3},{32,0,16,24,30,29},{0,32,56,52,48,51},{0,48,32,40,46,45},{32,0,24,16,20,23},{32,16,8,0,6,5},{0,48,40,32,36,39},{0,32,56,48,54,53},{32,16,0,8,12,15},{32,0,24,20,18,17},{0,32,48,56,60,63},{0,48,40,36,34,32},{32,0,16,24,28,31},{32,16,0,12,10,8},{0,48,32,40,44,47},{0,32,48,60,58,56},{32,16,8,0,4,7},{32,0,16,28,24,26},{0,32,56,48,52,55},{0,48,32,44,40,42},{32,0,24,20,16,19},{32,16,8,4,0,2},{0,48,40,36,32,35},{0,32,56,52,48,50},{32,16,0,12,8,11},{32,0,24,16,20,22},{0,32,48,60,56,59},{0,48,40,32,36,38},{32,0,16,28,26,25},{32,16,0,8,12,14},{0,48,32,44,42,41},{0,32,48,56,60,62},{32,16,8,4,2,0},{32,0,16,24,28,30},{0,32,56,52,50,48},{0,48,32,40,44,46},{32,0,24,16,22,20},{32,16,8,0,4,6},{0,48,40,32,38,36},{0,32,56,48,52,54},{32,16,0,8,14,12},{32,0,24,20,16,18},{0,32,48,56,62,60},{0,48,40,36,32,34},{32,0,16,24,30,28},{32,16,0,12,8,10},{0,48,32,40,46,44},{0,32,48,60,56,58},{32,16,8,0,6,4},{32,0,16,28,26,24},{0,32,56,48,54,52},{0,48,32,44,42,40},{32,0,24,20,18,16}};
const int recv_64[64][6] = {{0,0,0,0,0,0},{32,32,32,32,32,32},{32,48,48,48,48,48},{0,0,8,8,8,8},{0,16,16,20,20,20},{32,48,56,56,56,56},{32,32,32,36,36,36},{0,16,24,24,26,26},{0,0,8,12,12,12},{32,32,40,40,42,42},{32,48,56,60,60,60},{0,0,0,0,2,2},{0,16,24,28,28,28},{32,48,48,48,50,50},{32,32,40,44,44,44},{0,16,16,20,22,22},{0,0,0,4,4,4},{32,32,32,36,38,38},{32,48,48,52,52,52},{0,0,8,12,14,14},{0,16,16,16,16,16},{32,48,56,60,62,62},{32,32,32,32,32,33},{0,16,24,28,30,30},{0,0,8,8,8,9},{32,32,40,44,46,46},{32,48,56,56,56,57},{0,0,0,4,6,6},{0,16,24,24,26,27},{32,48,48,52,54,54},{32,32,40,40,42,43},{0,16,16,16,18,18},{0,0,0,0,2,3},{32,32,32,32,34,34},{32,48,48,48,50,51},{0,0,8,8,10,10},{0,16,16,20,22,23},{32,48,56,56,58,58},{32,32,32,36,38,39},{0,16,24,24,24,24},{0,0,8,12,14,15},{32,32,40,40,40,40},{32,48,56,60,62,63},{0,0,0,0,0,1},{0,16,24,28,30,31},{32,48,48,48,48,49},{32,32,40,44,46,47},{0,16,16,20,20,21},{0,0,0,4,6,7},{32,32,32,36,36,37},{32,48,48,52,54,55},{0,0,8,12,12,13},{0,16,16,16,18,19},{32,48,56,60,60,61},{32,32,32,32,34,35},{0,16,24,28,28,29},{0,0,8,8,10,11},{32,32,40,44,44,45},{32,48,56,56,58,59},{0,0,0,4,4,5},{0,16,24,24,24,25},{32,48,48,52,52,53},{32,32,40,40,40,41},{0,16,16,16,16,17}};

const int send_128[128][7] = {{64,32,16,8,4,2,1},{0,96,80,72,68,66,65},{0,64,112,104,100,98,97},{64,32,0,24,20,18,17},{64,0,48,32,44,42,41},{0,64,96,120,116,114,113},{0,96,80,64,76,74,73},{64,0,32,56,48,54,53},{64,32,0,16,28,26,25},{0,96,64,88,80,86,85},{0,64,96,112,124,122,121},{64,32,16,8,0,6,5},{64,0,32,48,60,56,59},{0,64,112,104,96,102,101},{0,96,64,80,92,88,91},{64,0,48,32,40,46,45},{64,32,16,0,12,8,11},{0,96,80,64,72,78,77},{0,64,112,96,108,104,107},{64,32,0,16,24,30,29},{64,0,48,40,36,32,35},{0,64,96,112,120,126,125},{0,96,80,72,68,64,67},{64,0,32,48,56,62,61},{64,32,0,24,20,16,19},{0,96,64,80,88,94,93},{0,64,96,120,116,112,115},{64,32,16,0,8,14,13},{64,0,32,56,48,52,55},{0,64,112,96,104,110,109},{0,96,64,88,80,84,87},{64,0,48,40,32,38,37},{64,32,16,8,0,4,7},{0,96,80,72,64,70,69},{0,64,112,104,96,100,103},{64,32,0,24,16,22,21},{64,0,48,32,40,44,47},{0,64,96,120,112,118,117},{0,96,80,64,72,76,79},{64,0,32,56,52,50,49},{64,32,0,16,24,28,31},{0,96,64,88,84,82,81},{0,64,96,112,120,124,127},{64,32,16,8,4,2,0},{64,0,32,48,56,60,63},{0,64,112,104,100,98,96},{0,96,64,80,88,92,95},{64,0,48,32,44,42,40},{64,32,16,0,8,12,15},{0,96,80,64,76,74,72},{0,64,112,96,104,108,111},{64,32,0,16,28,26,24},{64,0,48,40,32,36,39},{0,64,96,112,124,122,120},{0,96,80,72,64,68,71},{64,0,32,48,60,56,58},{64,32,0,24,16,20,23},{0,96,64,80,92,88,90},{0,64,96,120,112,116,119},{64,32,16,0,12,8,10},{64,0,32,56,52,48,51},{0,64,112,96,108,104,106},{0,96,64,88,84,80,83},{64,0,48,40,36,32,34},{64,32,16,8,4,0,3},{0,96,80,72,68,64,66},{0,64,112,104,100,96,99},{64,32,0,24,20,16,18},{64,0,48,32,44,40,43},{0,64,96,120,116,112,114},{0,96,80,64,76,72,75},{64,0,32,56,48,52,54},{64,32,0,16,28,24,27},{0,96,64,88,80,84,86},{0,64,96,112,124,120,123},{64,32,16,8,0,4,6},{64,0,32,48,60,58,57},{0,64,112,104,96,100,102},{0,96,64,80,92,90,89},{64,0,48,32,40,44,46},{64,32,16,0,12,10,9},{0,96,80,64,72,76,78},{0,64,112,96,108,106,105},{64,32,0,16,24,28,30},{64,0,48,40,36,34,33},{0,64,96,112,120,124,126},{0,96,80,72,68,66,64},{64,0,32,48,56,60,62},{64,32,0,24,20,18,16},{0,96,64,80,88,92,94},{0,64,96,120,116,114,112},{64,32,16,0,8,12,14},{64,0,32,56,48,54,52},{0,64,112,96,104,108,110},{0,96,64,88,80,86,84},{64,0,48,40,32,36,38},{64,32,16,8,0,6,4},{0,96,80,72,64,68,70},{0,64,112,104,96,102,100},{64,32,0,24,16,20,22},{64,0,48,32,40,46,44},{0,64,96,120,112,116,118},{0,96,80,64,72,78,76},{64,0,32,56,52,48,50},{64,32,0,16,24,30,28},{0,96,64,88,84,80,82},{0,64,96,112,120,126,124},{64,32,16,8,4,0,2},{64,0,32,48,56,62,60},{0,64,112,104,100,96,98},{0,96,64,80,88,94,92},{64,0,48,32,44,40,42},{64,32,16,0,8,14,12},{0,96,80,64,76,72,74},{0,64,112,96,104,110,108},{64,32,0,16,28,24,26},{64,0,48,40,32,38,36},{0,64,96,112,124,120,122},{0,96,80,72,64,70,68},{64,0,32,48,60,58,56},{64,32,0,24,16,22,20},{0,96,64,80,92,90,88},{0,64,96,120,112,118,116},{64,32,16,0,12,10,8},{64,0,32,56,52,50,48},{0,64,112,96,108,106,104},{0,96,64,88,84,82,80},{64,0,48,40,36,34,32}};
const int recv_128[128][7] = {{0,0,0,0,0,0,0},{64,64,64,64,64,64,64},{64,96,96,96,96,96,96},{0,0,16,16,16,16,16},{0,32,32,40,40,40,40},{64,96,112,112,112,112,112},{64,64,64,72,72,72,72},{0,32,48,48,52,52,52},{0,0,16,24,24,24,24},{64,64,80,80,84,84,84},{64,96,112,120,120,120,120},{0,0,0,0,4,4,4},{0,32,48,56,56,58,58},{64,96,96,96,100,100,100},{64,64,80,88,88,90,90},{0,32,32,40,44,44,44},{0,0,0,8,8,10,10},{64,64,64,72,76,76,76},{64,96,96,104,104,106,106},{0,0,16,24,28,28,28},{0,32,32,32,32,34,34},{64,96,112,120,124,124,124},{64,64,64,64,64,66,66},{0,32,48,56,60,60,60},{0,0,16,16,16,18,18},{64,64,80,88,92,92,92},{64,96,112,112,112,114,114},{0,0,0,8,12,12,12},{0,32,48,48,52,54,54},{64,96,96,104,108,108,108},{64,64,80,80,84,86,86},{0,32,32,32,36,36,36},{0,0,0,0,4,6,6},{64,64,64,64,68,68,68},{64,96,96,96,100,102,102},{0,0,16,16,20,20,20},{0,32,32,40,44,46,46},{64,96,112,112,116,116,116},{64,64,64,72,76,78,78},{0,32,48,48,48,48,48},{0,0,16,24,28,30,30},{64,64,80,80,80,80,80},{64,96,112,120,124,126,126},{0,0,0,0,0,0,1},{0,32,48,56,60,62,62},{64,96,96,96,96,96,97},{64,64,80,88,92,94,94},{0,32,32,40,40,40,41},{0,0,0,8,12,14,14},{64,64,64,72,72,72,73},{64,96,96,104,108,110,110},{0,0,16,24,24,24,25},{0,32,32,32,36,38,38},{64,96,112,120,120,120,121},{64,64,64,64,68,70,70},{0,32,48,56,56,58,59},{0,0,16,16,20,22,22},{64,64,80,88,88,90,91},{64,96,112,112,116,118,118},{0,0,0,8,8,10,11},{0,32,48,48,48,50,50},{64,96,96,104,104,106,107},{64,64,80,80,80,82,82},{0,32,32,32,32,34,35},{0,0,0,0,0,2,2},{64,64,64,64,64,66,67},{64,96,96,96,96,98,98},{0,0,16,16,16,18,19},{0,32,32,40,40,42,42},{64,96,112,112,112,114,115},{64,64,64,72,72,74,74},{0,32,48,48,52,54,55},{0,0,16,24,24,26,26},{64,64,80,80,84,86,87},{64,96,112,120,120,122,122},{0,0,0,0,4,6,7},{0,32,48,56,56,56,56},{64,96,96,96,100,102,103},{64,64,80,88,88,88,88},{0,32,32,40,44,46,47},{0,0,0,8,8,8,8},{64,64,64,72,76,78,79},{64,96,96,104,104,104,104},{0,0,16,24,28,30,31},{0,32,32,32,32,32,32},{64,96,112,120,124,126,127},{64,64,64,64,64,64,65},{0,32,48,56,60,62,63},{0,0,16,16,16,16,17},{64,64,80,88,92,94,95},{64,96,112,112,112,112,113},{0,0,0,8,12,14,15},{0,32,48,48,52,52,53},{64,96,96,104,108,110,111},{64,64,80,80,84,84,85},{0,32,32,32,36,38,39},{0,0,0,0,4,4,5},{64,64,64,64,68,70,71},{64,96,96,96,100,100,101},{0,0,16,16,20,22,23},{0,32,32,40,44,44,45},{64,96,112,112,116,118,119},{64,64,64,72,76,76,77},{0,32,48,48,48,50,51},{0,0,16,24,28,28,29},{64,64,80,80,80,82,83},{64,96,112,120,124,124,125},{0,0,0,0,0,2,3},{0,32,48,56,60,60,61},{64,96,96,96,96,98,99},{64,64,80,88,92,92,93},{0,32,32,40,40,42,43},{0,0,0,8,12,12,13},{64,64,64,72,72,74,75},{64,96,96,104,108,108,109},{0,0,16,24,24,26,27},{0,32,32,32,36,36,37},{64,96,112,120,120,122,123},{64,64,64,64,68,68,69},{0,32,48,56,56,56,57},{0,0,16,16,20,20,21},{64,64,80,88,88,88,89},{64,96,112,112,116,116,117},{0,0,0,8,8,8,9},{0,32,48,48,48,48,49},{64,96,96,104,104,104,105},{64,64,80,80,80,80,81},{0,32,32,32,32,32,33}};

const int send_256[256][8] = {{128,64,32,16,8,4,2,1},{0,192,160,144,136,132,130,129},{0,128,224,208,200,196,194,193},{128,64,0,48,40,36,34,33},{128,0,96,64,88,84,82,81},{0,128,192,240,232,228,226,225},{0,192,160,128,152,148,146,145},{128,0,64,112,96,108,106,105},{128,64,0,32,56,52,50,49},{0,192,128,176,160,172,170,169},{0,128,192,224,248,244,242,241},{128,64,32,16,0,12,10,9},{128,0,64,96,120,112,118,117},{0,128,224,208,192,204,202,201},{0,192,128,160,184,176,182,181},{128,0,96,64,80,92,90,89},{128,64,32,0,24,16,22,21},{0,192,160,128,144,156,154,153},{0,128,224,192,216,208,214,213},{128,64,0,32,48,60,58,57},{128,0,96,80,72,64,70,69},{0,128,192,224,240,252,250,249},{0,192,160,144,136,128,134,133},{128,0,64,96,112,124,120,123},{128,64,0,48,40,32,38,37},{0,192,128,160,176,188,184,187},{0,128,192,240,232,224,230,229},{128,64,32,0,16,28,24,27},{128,0,64,112,96,104,110,109},{0,128,224,192,208,220,216,219},{0,192,128,176,160,168,174,173},{128,0,96,80,64,76,72,75},{128,64,32,16,0,8,14,13},{0,192,160,144,128,140,136,139},{0,128,224,208,192,200,206,205},{128,64,0,48,32,44,40,43},{128,0,96,64,80,88,94,93},{0,128,192,240,224,236,232,235},{0,192,160,128,144,152,158,157},{128,0,64,112,104,100,96,99},{128,64,0,32,48,56,62,61},{0,192,128,176,168,164,160,163},{0,128,192,224,240,248,254,253},{128,64,32,16,8,4,0,3},{128,0,64,96,112,120,126,125},{0,128,224,208,200,196,192,195},{0,192,128,160,176,184,190,189},{128,0,96,64,88,84,80,83},{128,64,32,0,16,24,30,29},{0,192,160,128,152,148,144,147},{0,128,224,192,208,216,222,221},{128,64,0,32,56,52,48,51},{128,0,96,80,64,72,78,77},{0,128,192,224,248,244,240,243},{0,192,160,144,128,136,142,141},{128,0,64,96,120,112,116,119},{128,64,0,48,32,40,46,45},{0,192,128,160,184,176,180,183},{0,128,192,240,224,232,238,237},{128,64,32,0,24,16,20,23},{128,0,64,112,104,96,102,101},{0,128,224,192,216,208,212,215},{0,192,128,176,168,160,166,165},{128,0,96,80,72,64,68,71},{128,64,32,16,8,0,6,5},{0,192,160,144,136,128,132,135},{0,128,224,208,200,192,198,197},{128,64,0,48,40,32,36,39},{128,0,96,64,88,80,86,85},{0,128,192,240,232,224,228,231},{0,192,160,128,152,144,150,149},{128,0,64,112,96,104,108,111},{128,64,0,32,56,48,54,53},{0,192,128,176,160,168,172,175},{0,128,192,224,248,240,246,245},{128,64,32,16,0,8,12,15},{128,0,64,96,120,116,114,113},{0,128,224,208,192,200,204,207},{0,192,128,160,184,180,178,177},{128,0,96,64,80,88,92,95},{128,64,32,0,24,20,18,17},{0,192,160,128,144,152,156,159},{0,128,224,192,216,212,210,209},{128,64,0,32,48,56,60,63},{128,0,96,80,72,68,66,65},{0,128,192,224,240,248,252,255},{0,192,160,144,136,132,130,128},{128,0,64,96,112,120,124,127},{128,64,0,48,40,36,34,32},{0,192,128,160,176,184,188,191},{0,128,192,240,232,228,226,224},{128,64,32,0,16,24,28,31},{128,0,64,112,96,108,106,104},{0,128,224,192,208,216,220,223},{0,192,128,176,160,172,170,168},{128,0,96,80,64,72,76,79},{128,64,32,16,0,12,10,8},{0,192,160,144,128,136,140,143},{0,128,224,208,192,204,202,200},{128,64,0,48,32,40,44,47},{128,0,96,64,80,92,90,88},{0,128,192,240,224,232,236,239},{0,192,160,128,144,156,154,152},{128,0,64,112,104,96,100,103},{128,64,0,32,48,60,58,56},{0,192,128,176,168,160,164,167},{0,128,192,224,240,252,250,248},{128,64,32,16,8,0,4,7},{128,0,64,96,112,124,120,122},{0,128,224,208,200,192,196,199},{0,192,128,160,176,188,184,186},{128,0,96,64,88,80,84,87},{128,64,32,0,16,28,24,26},{0,192,160,128,152,144,148,151},{0,128,224,192,208,220,216,218},{128,64,0,32,56,48,52,55},{128,0,96,80,64,76,72,74},{0,128,192,224,248,240,244,247},{0,192,160,144,128,140,136,138},{128,0,64,96,120,116,112,115},{128,64,0,48,32,44,40,42},{0,192,128,160,184,180,176,179},{0,128,192,240,224,236,232,234},{128,64,32,0,24,20,16,19},{128,0,64,112,104,100,96,98},{0,128,224,192,216,212,208,211},{0,192,128,176,168,164,160,162},{128,0,96,80,72,68,64,67},{128,64,32,16,8,4,0,2},{0,192,160,144,136,132,128,131},{0,128,224,208,200,196,192,194},{128,64,0,48,40,36,32,35},{128,0,96,64,88,84,80,82},{0,128,192,240,232,228,224,227},{0,192,160,128,152,148,144,146},{128,0,64,112,96,108,104,107},{128,64,0,32,56,52,48,50},{0,192,128,176,160,172,168,171},{0,128,192,224,248,244,240,242},{128,64,32,16,0,12,8,11},{128,0,64,96,120,112,116,118},{0,128,224,208,192,204,200,203},{0,192,128,160,184,176,180,182},{128,0,96,64,80,92,88,91},{128,64,32,0,24,16,20,22},{0,192,160,128,144,156,152,155},{0,128,224,192,216,208,212,214},{128,64,0,32,48,60,56,59},{128,0,96,80,72,64,68,70},{0,128,192,224,240,252,248,251},{0,192,160,144,136,128,132,134},{128,0,64,96,112,124,122,121},{128,64,0,48,40,32,36,38},{0,192,128,160,176,188,186,185},{0,128,192,240,232,224,228,230},{128,64,32,0,16,28,26,25},{128,0,64,112,96,104,108,110},{0,128,224,192,208,220,218,217},{0,192,128,176,160,168,172,174},{128,0,96,80,64,76,74,73},{128,64,32,16,0,8,12,14},{0,192,160,144,128,140,138,137},{0,128,224,208,192,200,204,206},{128,64,0,48,32,44,42,41},{128,0,96,64,80,88,92,94},{0,128,192,240,224,236,234,233},{0,192,160,128,144,152,156,158},{128,0,64,112,104,100,98,97},{128,64,0,32,48,56,60,62},{0,192,128,176,168,164,162,161},{0,128,192,224,240,248,252,254},{128,64,32,16,8,4,2,0},{128,0,64,96,112,120,124,126},{0,128,224,208,200,196,194,192},{0,192,128,160,176,184,188,190},{128,0,96,64,88,84,82,80},{128,64,32,0,16,24,28,30},{0,192,160,128,152,148,146,144},{0,128,224,192,208,216,220,222},{128,64,0,32,56,52,50,48},{128,0,96,80,64,72,76,78},{0,128,192,224,248,244,242,240},{0,192,160,144,128,136,140,142},{128,0,64,96,120,112,118,116},{128,64,0,48,32,40,44,46},{0,192,128,160,184,176,182,180},{0,128,192,240,224,232,236,238},{128,64,32,0,24,16,22,20},{128,0,64,112,104,96,100,102},{0,128,224,192,216,208,214,212},{0,192,128,176,168,160,164,166},{128,0,96,80,72,64,70,68},{128,64,32,16,8,0,4,6},{0,192,160,144,136,128,134,132},{0,128,224,208,200,192,196,198},{128,64,0,48,40,32,38,36},{128,0,96,64,88,80,84,86},{0,128,192,240,232,224,230,228},{0,192,160,128,152,144,148,150},{128,0,64,112,96,104,110,108},{128,64,0,32,56,48,52,54},{0,192,128,176,160,168,174,172},{0,128,192,224,248,240,244,246},{128,64,32,16,0,8,14,12},{128,0,64,96,120,116,112,114},{0,128,224,208,192,200,206,204},{0,192,128,160,184,180,176,178},{128,0,96,64,80,88,94,92},{128,64,32,0,24,20,16,18},{0,192,160,128,144,152,158,156},{0,128,224,192,216,212,208,210},{128,64,0,32,48,56,62,60},{128,0,96,80,72,68,64,66},{0,128,192,224,240,248,254,252},{0,192,160,144,136,132,128,130},{128,0,64,96,112,120,126,124},{128,64,0,48,40,36,32,34},{0,192,128,160,176,184,190,188},{0,128,192,240,232,228,224,226},{128,64,32,0,16,24,30,28},{128,0,64,112,96,108,104,106},{0,128,224,192,208,216,222,220},{0,192,128,176,160,172,168,170},{128,0,96,80,64,72,78,76},{128,64,32,16,0,12,8,10},{0,192,160,144,128,136,142,140},{0,128,224,208,192,204,200,202},{128,64,0,48,32,40,46,44},{128,0,96,64,80,92,88,90},{0,128,192,240,224,232,238,236},{0,192,160,128,144,156,152,154},{128,0,64,112,104,96,102,100},{128,64,0,32,48,60,56,58},{0,192,128,176,168,160,166,164},{0,128,192,224,240,252,248,250},{128,64,32,16,8,0,6,4},{128,0,64,96,112,124,122,120},{0,128,224,208,200,192,198,196},{0,192,128,160,176,188,186,184},{128,0,96,64,88,80,86,84},{128,64,32,0,16,28,26,24},{0,192,160,128,152,144,150,148},{0,128,224,192,208,220,218,216},{128,64,0,32,56,48,54,52},{128,0,96,80,64,76,74,72},{0,128,192,224,248,240,246,244},{0,192,160,144,128,140,138,136},{128,0,64,96,120,116,114,112},{128,64,0,48,32,44,42,40},{0,192,128,160,184,180,178,176},{0,128,192,240,224,236,234,232},{128,64,32,0,24,20,18,16},{128,0,64,112,104,100,98,96},{0,128,224,192,216,212,210,208},{0,192,128,176,168,164,162,160},{128,0,96,80,72,68,66,64}};
const int recv_256[256][8] = {{0,0,0,0,0,0,0,0},{128,128,128,128,128,128,128,128},{128,192,192,192,192,192,192,192},{0,0,32,32,32,32,32,32},{0,64,64,80,80,80,80,80},{128,192,224,224,224,224,224,224},{128,128,128,144,144,144,144,144},{0,64,96,96,104,104,104,104},{0,0,32,48,48,48,48,48},{128,128,160,160,168,168,168,168},{128,192,224,240,240,240,240,240},{0,0,0,0,8,8,8,8},{0,64,96,112,112,116,116,116},{128,192,192,192,200,200,200,200},{128,128,160,176,176,180,180,180},{0,64,64,80,88,88,88,88},{0,0,0,16,16,20,20,20},{128,128,128,144,152,152,152,152},{128,192,192,208,208,212,212,212},{0,0,32,48,56,56,56,56},{0,64,64,64,64,68,68,68},{128,192,224,240,248,248,248,248},{128,128,128,128,128,132,132,132},{0,64,96,112,120,120,122,122},{0,0,32,32,32,36,36,36},{128,128,160,176,184,184,186,186},{128,192,224,224,224,228,228,228},{0,0,0,16,24,24,26,26},{0,64,96,96,104,108,108,108},{128,192,192,208,216,216,218,218},{128,128,160,160,168,172,172,172},{0,64,64,64,72,72,74,74},{0,0,0,0,8,12,12,12},{128,128,128,128,136,136,138,138},{128,192,192,192,200,204,204,204},{0,0,32,32,40,40,42,42},{0,64,64,80,88,92,92,92},{128,192,224,224,232,232,234,234},{128,128,128,144,152,156,156,156},{0,64,96,96,96,96,98,98},{0,0,32,48,56,60,60,60},{128,128,160,160,160,160,162,162},{128,192,224,240,248,252,252,252},{0,0,0,0,0,0,2,2},{0,64,96,112,120,124,124,124},{128,192,192,192,192,192,194,194},{128,128,160,176,184,188,188,188},{0,64,64,80,80,80,82,82},{0,0,0,16,24,28,28,28},{128,128,128,144,144,144,146,146},{128,192,192,208,216,220,220,220},{0,0,32,48,48,48,50,50},{0,64,64,64,72,76,76,76},{128,192,224,240,240,240,242,242},{128,128,128,128,136,140,140,140},{0,64,96,112,112,116,118,118},{0,0,32,32,40,44,44,44},{128,128,160,176,176,180,182,182},{128,192,224,224,232,236,236,236},{0,0,0,16,16,20,22,22},{0,64,96,96,96,100,100,100},{128,192,192,208,208,212,214,214},{128,128,160,160,160,164,164,164},{0,64,64,64,64,68,70,70},{0,0,0,0,0,4,4,4},{128,128,128,128,128,132,134,134},{128,192,192,192,192,196,196,196},{0,0,32,32,32,36,38,38},{0,64,64,80,80,84,84,84},{128,192,224,224,224,228,230,230},{128,128,128,144,144,148,148,148},{0,64,96,96,104,108,110,110},{0,0,32,48,48,52,52,52},{128,128,160,160,168,172,174,174},{128,192,224,240,240,244,244,244},{0,0,0,0,8,12,14,14},{0,64,96,112,112,112,112,112},{128,192,192,192,200,204,206,206},{128,128,160,176,176,176,176,176},{0,64,64,80,88,92,94,94},{0,0,0,16,16,16,16,16},{128,128,128,144,152,156,158,158},{128,192,192,208,208,208,208,208},{0,0,32,48,56,60,62,62},{0,64,64,64,64,64,64,64},{128,192,224,240,248,252,254,254},{128,128,128,128,128,128,128,129},{0,64,96,112,120,124,126,126},{0,0,32,32,32,32,32,33},{128,128,160,176,184,188,190,190},{128,192,224,224,224,224,224,225},{0,0,0,16,24,28,30,30},{0,64,96,96,104,104,104,105},{128,192,192,208,216,220,222,222},{128,128,160,160,168,168,168,169},{0,64,64,64,72,76,78,78},{0,0,0,0,8,8,8,9},{128,128,128,128,136,140,142,142},{128,192,192,192,200,200,200,201},{0,0,32,32,40,44,46,46},{0,64,64,80,88,88,88,89},{128,192,224,224,232,236,238,238},{128,128,128,144,152,152,152,153},{0,64,96,96,96,100,102,102},{0,0,32,48,56,56,56,57},{128,128,160,160,160,164,166,166},{128,192,224,240,248,248,248,249},{0,0,0,0,0,4,6,6},{0,64,96,112,120,120,122,123},{128,192,192,192,192,196,198,198},{128,128,160,176,184,184,186,187},{0,64,64,80,80,84,86,86},{0,0,0,16,24,24,26,27},{128,128,128,144,144,148,150,150},{128,192,192,208,216,216,218,219},{0,0,32,48,48,52,54,54},{0,64,64,64,72,72,74,75},{128,192,224,240,240,244,246,246},{128,128,128,128,136,136,138,139},{0,64,96,112,112,112,114,114},{0,0,32,32,40,40,42,43},{128,128,160,176,176,176,178,178},{128,192,224,224,232,232,234,235},{0,0,0,16,16,16,18,18},{0,64,96,96,96,96,98,99},{128,192,192,208,208,208,210,210},{128,128,160,160,160,160,162,163},{0,64,64,64,64,64,66,66},{0,0,0,0,0,0,2,3},{128,128,128,128,128,128,130,130},{128,192,192,192,192,192,194,195},{0,0,32,32,32,32,34,34},{0,64,64,80,80,80,82,83},{128,192,224,224,224,224,226,226},{128,128,128,144,144,144,146,147},{0,64,96,96,104,104,106,106},{0,0,32,48,48,48,50,51},{128,128,160,160,168,168,170,170},{128,192,224,240,240,240,242,243},{0,0,0,0,8,8,10,10},{0,64,96,112,112,116,118,119},{128,192,192,192,200,200,202,202},{128,128,160,176,176,180,182,183},{0,64,64,80,88,88,90,90},{0,0,0,16,16,20,22,23},{128,128,128,144,152,152,154,154},{128,192,192,208,208,212,214,215},{0,0,32,48,56,56,58,58},{0,64,64,64,64,68,70,71},{128,192,224,240,248,248,250,250},{128,128,128,128,128,132,134,135},{0,64,96,112,120,120,120,120},{0,0,32,32,32,36,38,39},{128,128,160,176,184,184,184,184},{128,192,224,224,224,228,230,231},{0,0,0,16,24,24,24,24},{0,64,96,96,104,108,110,111},{128,192,192,208,216,216,216,216},{128,128,160,160,168,172,174,175},{0,64,64,64,72,72,72,72},{0,0,0,0,8,12,14,15},{128,128,128,128,136,136,136,136},{128,192,192,192,200,204,206,207},{0,0,32,32,40,40,40,40},{0,64,64,80,88,92,94,95},{128,192,224,224,232,232,232,232},{128,128,128,144,152,156,158,159},{0,64,96,96,96,96,96,96},{0,0,32,48,56,60,62,63},{128,128,160,160,160,160,160,160},{128,192,224,240,248,252,254,255},{0,0,0,0,0,0,0,1},{0,64,96,112,120,124,126,127},{128,192,192,192,192,192,192,193},{128,128,160,176,184,188,190,191},{0,64,64,80,80,80,80,81},{0,0,0,16,24,28,30,31},{128,128,128,144,144,144,144,145},{128,192,192,208,216,220,222,223},{0,0,32,48,48,48,48,49},{0,64,64,64,72,76,78,79},{128,192,224,240,240,240,240,241},{128,128,128,128,136,140,142,143},{0,64,96,112,112,116,116,117},{0,0,32,32,40,44,46,47},{128,128,160,176,176,180,180,181},{128,192,224,224,232,236,238,239},{0,0,0,16,16,20,20,21},{0,64,96,96,96,100,102,103},{128,192,192,208,208,212,212,213},{128,128,160,160,160,164,166,167},{0,64,64,64,64,68,68,69},{0,0,0,0,0,4,6,7},{128,128,128,128,128,132,132,133},{128,192,192,192,192,196,198,199},{0,0,32,32,32,36,36,37},{0,64,64,80,80,84,86,87},{128,192,224,224,224,228,228,229},{128,128,128,144,144,148,150,151},{0,64,96,96,104,108,108,109},{0,0,32,48,48,52,54,55},{128,128,160,160,168,172,172,173},{128,192,224,240,240,244,246,247},{0,0,0,0,8,12,12,13},{0,64,96,112,112,112,114,115},{128,192,192,192,200,204,204,205},{128,128,160,176,176,176,178,179},{0,64,64,80,88,92,92,93},{0,0,0,16,16,16,18,19},{128,128,128,144,152,156,156,157},{128,192,192,208,208,208,210,211},{0,0,32,48,56,60,60,61},{0,64,64,64,64,64,66,67},{128,192,224,240,248,252,252,253},{128,128,128,128,128,128,130,131},{0,64,96,112,120,124,124,125},{0,0,32,32,32,32,34,35},{128,128,160,176,184,188,188,189},{128,192,224,224,224,224,226,227},{0,0,0,16,24,28,28,29},{0,64,96,96,104,104,106,107},{128,192,192,208,216,220,220,221},{128,128,160,160,168,168,170,171},{0,64,64,64,72,76,76,77},{0,0,0,0,8,8,10,11},{128,128,128,128,136,140,140,141},{128,192,192,192,200,200,202,203},{0,0,32,32,40,44,44,45},{0,64,64,80,88,88,90,91},{128,192,224,224,232,236,236,237},{128,128,128,144,152,152,154,155},{0,64,96,96,96,100,100,101},{0,0,32,48,56,56,58,59},{128,128,160,160,160,164,164,165},{128,192,224,240,248,248,250,251},{0,0,0,0,0,4,4,5},{0,64,96,112,120,120,120,121},{128,192,192,192,192,196,196,197},{128,128,160,176,184,184,184,185},{0,64,64,80,80,84,84,85},{0,0,0,16,24,24,24,25},{128,128,128,144,144,148,148,149},{128,192,192,208,216,216,216,217},{0,0,32,48,48,52,52,53},{0,64,64,64,72,72,72,73},{128,192,224,240,240,244,244,245},{128,128,128,128,136,136,136,137},{0,64,96,112,112,112,112,113},{0,0,32,32,40,40,40,41},{128,128,160,176,176,176,176,177},{128,192,224,224,232,232,232,233},{0,0,0,16,16,16,16,17},{0,64,96,96,96,96,96,97},{128,192,192,208,208,208,208,209},{128,128,160,160,160,160,160,161},{0,64,64,64,64,64,64,65}};

const void* static_send_bitmaps[] = { NULL, send_2, send_4, send_8, send_16, send_32, send_64, send_128, send_256 };
const void* static_recv_bitmaps[] = { NULL, recv_2, recv_4, recv_8, recv_16, recv_32, recv_64, recv_128, recv_256 };
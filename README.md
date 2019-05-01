<h3>FastSent</h3>

During the <b>Deep Learning Udacity Nanodegree</b>, we were asked to implement a neuralnet with 1 hidden layer in pure Python. This project is in pure C.

<b>Code is not clean and this project is actually a playground for me to better understand concepts and technology limitations</b> but still had better metrics than LSTM/CNN approaches tried on the same dataset.

Key implementation info:
1) Pure C even for Backpropagation and Stochastic Gradient Descent
2) Multi-Threaded
3) NN Weights saved/loaded
4) Early Stopping so no need to specify number of epochs
5) Learning Rate range test so no need to specify learning rate
6) Two sentiment outputs objectivity/subjectivity and polarity (measure and score)

<h3>Usage:</h3>
<pre>./fastsent_train TRAIN_FILE TEST_FILE</pre>

<h3>Sample TRAIN/TEST file (with greeglish)</h3>
<pre>0  gia emas tous prwhn em kapnistes em o kafes xwris tsigaro einai san kalo sex alla xwris na se exei gleipsei kalos men alla
-1  user h ka em dourou em kanei panta tis katallhles energeies einai apotelesmatikh kai kataplhktika ergatikh
0  antimetwpoi me kyrwseis osoi moirazontai kwdikous em netflix em url
-1  katadikh 0 ypallhlwn gia em kakopoihsh em trofimwn se idryma em paidiwn em me anaphria #politikgr url mesw tou xrhsth user
1  des tous orous tou diagwnismou edw url
-1  user tragikh etaireia anyparkth taxythta internet #badcustomerservice me anamonh 0 wres se anamonh syndeshs se user edw kai 0 wres
-1  o allos pshnei tsikna xamos kanei k hlektroniko em tsigaro em k mazi me to seiniko teixos fainetai ki autos ap to diasthma
-1  sta 0 eurw h timh tou paketou em tsigarwn em sth gallia to 0 url
0  perierges allages exoun ginei sth diamorfwsh tou genikou deikth me ton em ote em na pernaei 0 os pisw apo th eee thn alpha na exakontizetai
-1  dwsei o syriza blepe em dourou em 0 ekatommyria apo ta xrhmata mas twn forologoumenwn gia to ghpedo ths aek rwthse kanenan url ?
0  tha mpw em sta em em public em petwntas fylladia pou tha grafoun sto plaisio em ta em dinoun ola tsampa k tha feugw kanontas tsixlofouskes poios roubikwnas
0  user to exoun kai sth em noba em twra
0  prwth ellhnikh tainia mythoplasias robogirl gia thn ekpaideutikh rompotikh kai thn empeiria twn paidiwn pou asxolountai me auth parousiazei h em cosmote em
0  re gia peite ligo grhgora pou mporei na yparxei xexasmeno em filtraki em mes sto spt tha trelathw
0  poies proslhpseis anoixan sta souper market em metro em kai em my em em market em url
0  ergasia se ab basilopoulos kai em lidl em url
-1  pisteuw einai thema xronou na steiloun pisw ton epop oi tourkoi poso tha antexoun na akoune gia hlektroniko em tsigaro em url
-1  ftanw rio sta em diodia em blepw 0 0 ti egine lew akribynan pali oxi pali kyrie mou leei h kopelitsa ta teleutaia 0 0 xronia 0 0 eixan
0  to mono pou xreiazetai enas syndromhths ths em wind em vision gia na parakolouthhsei to athlhtiko ypertheama apo to smartphone h to tablet tou einai syndesh oi syndromhtes ths em wind em vision mporoun na epilexoun mia apo tis 0 forhtes syskeues pou mporoun na syndesoun sthn efarmogh em wind em vision kai na parakolouthoun erxetai h dynatothta parakolouthhshs twn kanaliwn novasports mesa apo th brabeumenh efarmogh gia forhtes syskeues ths em wind em vision
0  o papandreou h em kapnistria em syntrofos tou kai h entonh parousia tou stous delfous url</pre>



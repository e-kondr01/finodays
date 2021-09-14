--
-- PostgreSQL database dump
--

-- Dumped from database version 13.3 (Debian 13.3-1.pgdg100+1)
-- Dumped by pg_dump version 13.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: test_table; Type: TABLE; Schema: public; Owner: maindb_ml
--

CREATE TABLE public.test_table (
    mood integer NOT NULL,
    id integer NOT NULL,
    date_of_msg character varying(32),
    idk_what character varying(16),
    idk_what2 character varying(16),
    text_msg character varying(2048)
);


ALTER TABLE public.test_table OWNER TO maindb_ml;

--
-- Data for Name: test_table; Type: TABLE DATA; Schema: public; Owner: maindb_ml
--

COPY public.test_table (mood, id, date_of_msg, idk_what, idk_what2, text_msg) FROM stdin;
4	3	Mon May 11 03:17:40 UTC 2009	kindle2	tpryan	@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right.
4	4	Mon May 11 03:18:03 UTC 2009	kindle2	vcu451	Reading my kindle2...  Love it... Lee childs is good read.
4	5	Mon May 11 03:18:54 UTC 2009	kindle2	chadfu	Ok, first assesment of the #kindle2 ...it fucking rocks!!!
4	6	Mon May 11 03:19:04 UTC 2009	kindle2	SIX15	@kenburbary You'll love your Kindle2. I've had mine for a few months and never looked back. The new big one is huge! No need for remorse! :)
4	7	Mon May 11 03:21:41 UTC 2009	kindle2	yamarama	@mikefish  Fair enough. But i have the Kindle2 and I think it's perfect  :)
4	8	Mon May 11 03:22:00 UTC 2009	kindle2	GeorgeVHulme	@richardebaker no. it is too big. I'm quite happy with the Kindle2.
0	9	Mon May 11 03:22:30 UTC 2009	aig	Seth937	Fuck this economy. I hate aig and their non loan given asses.
4	10	Mon May 11 03:26:10 UTC 2009	jquery	dcostalis	Jquery is my new best friend.
4	11	Mon May 11 03:27:15 UTC 2009	twitter	PJ_King	Loves twitter
4	12	Mon May 11 03:29:20 UTC 2009	obama	mandanicole	how can you not love Obama? he makes jokes about himself.
2	13	Mon May 11 03:32:42 UTC 2009	obama	jpeb	Check this video out -- President Obama at the White House Correspondents' Dinner http://bit.ly/IMXUM
0	14	Mon May 11 03:32:48 UTC 2009	obama	kylesellers	@Karoli I firmly believe that Obama/Pelosi have ZERO desire to be civil.  It's a charade and a slogan, but they want to destroy conservatism
4	15	Mon May 11 03:33:38 UTC 2009	obama	theviewfans	House Correspondents dinner was last night whoopi, barbara &amp; sherri went, Obama got a standing ovation
4	16	Mon May 11 05:05:58 UTC 2009	nike	MumsFP	Watchin Espn..Jus seen this new Nike Commerical with a Puppet Lebron..sh*t was hilarious...LMAO!!!
0	17	Mon May 11 05:06:22 UTC 2009	nike	vincentx24x	dear nike, stop with the flywire. that shit is a waste of science. and ugly. love, @vincentx24x
4	18	Mon May 11 05:20:15 UTC 2009	lebron	cameronwylie	#lebron best athlete of our generation, if not all time (basketball related) I don't want to get into inter-sport debates about   __1/2
0	19	Mon May 11 05:20:28 UTC 2009	lebron	luv8242	I was talking to this guy last night and he was telling me that he is a die hard Spurs fan.  He also told me that he hates LeBron James.
4	20	Mon May 11 05:21:04 UTC 2009	lebron	mtgillikin	i love lebron. http://bit.ly/PdHur
0	21	Mon May 11 05:21:37 UTC 2009	lebron	ursecretdezire	@ludajuice Lebron is a Beast, but I'm still cheering 4 the A..til the end.
4	22	Mon May 11 05:21:45 UTC 2009	lebron	Native_01	@Pmillzz lebron IS THE BOSS
4	23	Mon May 11 05:22:03 UTC 2009	lebron	princezzcutz	@sketchbug Lebron is a hometown hero to me, lol I love the Lakers but let's go Cavs, lol
4	24	Mon May 11 05:22:12 UTC 2009	lebron	peterlikewhat	lebron and zydrunas are such an awesome duo
4	25	Mon May 11 05:22:37 UTC 2009	lebron	emceet	@wordwhizkid Lebron is a beast... nobody in the NBA comes even close.
4	26	Mon May 11 06:02:24 UTC 2009	iphone app	CocoSavanna	downloading apps for my iphone! So much fun :-) There literally is an app for just about anything.
4	33	Mon May 11 19:47:29 UTC 2009	visa	DreambigRadio	good news, just had a call from the Visa office, saying everything is fine.....what a relief! I am sick of scams out there! Stealing!
4	34	Mon May 11 19:49:21 UTC 2009	fredwilson	andrewwatson	http://twurl.nl/epkr4b - awesome come back from @biz (via @fredwilson)
4	35	Mon May 11 19:50:07 UTC 2009	fredwilson	fredwilson	In montreal for a long weekend of R&amp;R. Much needed.
4	46	Thu May 14 02:58:07 UTC 2009	"booz allen"	JoeSchueller	Booz Allen Hamilton has a bad ass homegrown social collaboration platform. Way cool!  #ttiv
4	47	Thu May 14 02:58:23 UTC 2009	"booz allen"	scottabel	[#MLUC09] Customer Innovation Award Winner: Booz Allen Hamilton -- http://ping.fm/c2hPP
4	49	Thu May 14 05:24:50 UTC 2009	40d	JustMe_D	@SoChi2 I current use the Nikon D90 and love it, but not as much as the Canon 40D/50D. I chose the D90 for the  video feature. My mistake.
2	50	Thu May 14 05:25:04 UTC 2009	40d	hiteshbagai	need suggestions for a good IR filter for my canon 40D ... got some? pls DM
2	117	Sat May 16 16:18:36 UTC 2009	google	Annimallover	@surfit: I just checked my google for my business- blip shows up as the second entry! Huh. Is that a good or ba... ? http://blip.fm/~6emhv
4	118	Sat May 16 16:19:04 UTC 2009	google	J_Holl	@phyreman9 Google is always a good place to look. Should've mentioned I worked on the Mustang w/ my Dad, @KimbleT.
0	119	Sat May 16 16:19:24 UTC 2009	google	vamsmack	Played with an android google phone. The slide out screen scares me I would break that fucker so fast. Still prefer my iPhone.
0	120	Sat May 16 16:25:41 UTC 2009	aig	schroncd	US planning to resume the military tribunals at Guantanamo Bay... only this time those on trial will be AIG execs and Chrysler debt holders
0	121	Sat May 16 22:42:07 UTC 2009	itchy	MarissaLeeD	omg so bored &amp; my tattoooos are so itchy!!  help! aha =)
0	122	Sat May 16 22:42:25 UTC 2009	itchy	robloposky	I'm itchy and miserable!
0	123	Sat May 16 22:42:44 UTC 2009	itchy	EdwinLValencia	@sekseemess no. I'm not itchy for now. Maybe later, lol.
4	124	Sat May 16 23:48:15 UTC 2009	stanford	imusicmash	RT @jessverr I love the nerdy Stanford human biology videos - makes me miss school. http://bit.ly/13t7NR
4	125	Sat May 16 23:58:34 UTC 2009	lyx	drewloewe	@spinuzzi: Has been a bit crazy, with steep learning curve, but LyX is really good for long docs. For anything shorter, it would be insane.
4	131	Sun May 17 15:05:03 UTC 2009	Danny Gokey	VickyTigger	I'm listening to "P.Y.T" by Danny Gokey &lt;3 &lt;3 &lt;3 Aww, he's so amazing. I &lt;3 him so much :)
4	132	Sun May 17 17:27:45 UTC 2009	sleep	babblyabbie	is going to sleep then on a bike ride:]
0	133	Sun May 17 17:27:49 UTC 2009	sleep	kisjoaquin	cant sleep... my tooth is aching.
0	134	Sun May 17 17:28:02 UTC 2009	sleep	Whacktackular	Blah, blah, blah same old same old. No plans today, going back to sleep I guess.
0	135	Sun May 17 17:29:50 UTC 2009	san francisco	Adrigonzo	glad i didnt do Bay to Breakers today, it's 1000 freaking degrees in San Francisco wtf
2	136	Sun May 17 17:30:19 UTC 2009	san francisco	sulu34	is in San Francisco at Bay to Breakers.
2	137	Sun May 17 17:30:23 UTC 2009	san francisco	schuyler	just landed at San Francisco
2	138	Sun May 17 17:30:56 UTC 2009	san francisco	MattBragoni	San Francisco today.  Any suggestions?
0	139	Sun May 17 17:32:00 UTC 2009	aig	KennyTRoland	?Obama Administration Must Stop Bonuses to AIG Ponzi Schemers ... http://bit.ly/2CUIg
0	140	Sun May 17 17:32:30 UTC 2009	aig	aMild	started to think that Citi is in really deep s&amp;^t. Are they gonna survive the turmoil or are they gonna be the next AIG?
0	141	Sun May 17 17:32:36 UTC 2009	aig	Trazor1	ShaunWoo hate'n on AiG
4	142	Sun May 17 17:35:17 UTC 2009	star trek	mimknits	@YarnThing you will not regret going to see Star Trek. It was AWESOME!
2	143	Sun May 17 17:35:28 UTC 2009	star trek	GeeRen	On my way to see Star Trek @ The Esquire.
2	144	Sun May 17 17:35:45 UTC 2009	star trek	checkyesjess	Going to see star trek soon with my dad.
0	145	Mon May 18 01:13:27 UTC 2009	Malcolm Gladwell	renano	annoying new trend on the internets:  people picking apart michael lewis and malcolm gladwell.  nobody wants to read that.
2	146	Mon May 18 01:14:41 UTC 2009	Malcolm Gladwell	kottkedotorg	Bill Simmons in conversation with Malcolm Gladwell http://bit.ly/j9o50
4	147	Mon May 18 01:14:47 UTC 2009	Malcolm Gladwell	davidm89	Highly recommend: http://tinyurl.com/HowDavidBeatsGoliath by Malcolm Gladwell
4	148	Mon May 18 01:15:17 UTC 2009	Malcolm Gladwell	livreal	Blink by malcolm gladwell amazing book and The tipping point!
4	149	Mon May 18 01:16:12 UTC 2009	Malcolm Gladwell	mikearosso	Malcolm Gladwell might be my new man crush
0	150	Mon May 18 01:18:03 UTC 2009	espn	wendy93639	omg. The commercials alone on ESPN are going to drive me nuts.
4	151	Mon May 18 03:11:06 UTC 2009	"twitter api"	ClayFranklin	@robmalon Playing with Twitter API sounds fun.  May need to take a class or find a new friend who like to generate results with API code.
2	152	Mon May 18 03:11:58 UTC 2009	"twitter api"	cURLTesting	playing with cURL and the Twitter API
4	153	Mon May 18 03:12:13 UTC 2009	"twitter api"	ringerdrop	Hello Twitter API ;)
2	154	Mon May 18 03:12:18 UTC 2009	"twitter api"	danserfaty	playing with Java and the Twitter API
0	155	Mon May 18 03:12:40 UTC 2009	"twitter api"	raykolbe	@morind45 Because the twitter api is slow and most client's aren't good.
0	156	Mon May 18 03:13:03 UTC 2009	yahoo	CarolineVilas	yahoo answers can be a butt sometimes
4	157	Mon May 18 05:07:16 UTC 2009	scrapbooking	rachelbegins	is scrapbooking with Nic =D
4	172	Tue May 19 16:20:46 UTC 2009	wolfram alpha	leedscentlib	RT @mashable: Five Things Wolfram Alpha Does Better (And Vastly Different) Than Google - http://bit.ly/6nSnR
4	173	Wed May 20 02:37:09 UTC 2009	nike	sportsgirl505	just changed my default pic to a Nike basketball cause bball is awesome!!!!!
2	174	Wed May 20 02:38:27 UTC 2009	nike	MatrixSystems	Nike owns NBA Playoffs ads w/ LeBron, Kobe, Carmelo? http://ow.ly/7Uiy  #Adidas #Billups #Howard  #Marketing #Branding
2	175	Wed May 20 02:38:45 UTC 2009	nike	cadburysgirl	'Next time, I'll call myself Nike'
2	176	Wed May 20 02:38:50 UTC 2009	nike	AddictedToFresh	New blog post: Nike SB Dunk Low Premium 'White Gum' http://tr.im/lOtT
0	177	Wed May 20 02:39:05 UTC 2009	nike	coreysmbpro	RT @SmartChickPDX: Was just told that Nike layoffs started today :-(
4	178	Wed May 20 02:39:28 UTC 2009	nike	PRolivia	Back when I worked for Nike we had one fav word : JUST DO IT! :)
4	179	Wed May 20 02:40:23 UTC 2009	nike	ErrantDreams	By the way, I'm totally inspired by this freaky Nike commercial: http://snurl.com/icgj9
2	192	Sat May 23 20:43:22 UTC 2009	weka	nikete	giving weka an app engine interface, using the bird strike data for the tests, the logo is a given.
2	194	Sun May 24 16:18:48 UTC 2009	50d	bigdigit	Brand New Canon EOS 50D 15MP DSLR Camera Canon 17-85mm IS Lens ...: Web Technology Thread, Brand New Canon EOS 5.. http://u.mavrev.com/5a3t
4	195	Sun May 24 16:19:04 UTC 2009	50d	justinbettman	Class... The 50d is supposed to come today :)
0	196	Sun May 24 20:48:14 UTC 2009	lambda calculus	read0	needs someone to explain lambda calculus to him! :(
0	197	Sun May 24 20:48:58 UTC 2009	lambda calculus	BAK3R	Took the Graduate Field Exam for Computer Science today.  Nothing makes you feel like more of an idiot than lambda calculus.
4	198	Sun May 24 20:49:52 UTC 2009	east palo alto	SLICKSPIT	SHOUT OUTS TO ALL EAST PALO ALTO FOR BEING IN THE BUILDIN KARIZMAKAZE 50CAL GTA! ALSO THANKS TO PROFITS OF DOOM UNIVERSAL HEMPZ CRACKA......
0	199	Sun May 24 20:50:32 UTC 2009	east palo alto	ckwright	@legalgeekery Yeahhhhhhhhh, I wouldn't really have lived in East Palo Alto if I could have avoided it.  I guess it's only for the summer.
4	204	Mon May 25 17:15:01 UTC 2009	stanford	souleaterjh	@accannis @edog1203 Great Stanford course. Thanks for making it available to the public! Really helpful and informative for starting off!
2	205	Mon May 25 17:15:24 UTC 2009	stanford	susangao	NVIDIA Names Stanford's Bill Dally Chief Scientist, VP Of Research http://bit.ly/Fvvg9
2	206	Mon May 25 17:15:43 UTC 2009	stanford	jimmy_chan2009	New blog post: Harvard Versus Stanford - Who Wins? http://bit.ly/MCoCo
4	207	Mon May 25 17:17:04 UTC 2009	lakers	a_fio	@ work til 6pm... lets go lakers!!!
0	208	Mon May 25 17:18:29 UTC 2009	north korea	luvslikepi	Damn you North Korea. http://bit.ly/KtMeQ
0	209	Mon May 25 17:19:05 UTC 2009	north korea	utsagrad123	Can we just go ahead and blow North Korea off the map already?
0	210	Mon May 25 17:19:14 UTC 2009	north korea	stabotage	North Korea, please cease this douchebaggery. China doesn't even like you anymore. http://bit.ly/NeHSl
0	211	Mon May 25 17:21:08 UTC 2009	pelosi	zed01	Why the hell is Pelosi in freakin China? and on whose dime?
0	212	Mon May 25 17:23:55 UTC 2009	bailout	funky_old_man	Are YOU burning more cash $$$ than Chrysler and GM? Stop the financial tsunami. Where "bailout" means taking a handout!
0	213	Mon May 25 17:25:34 UTC 2009	insects	malcozer	insects have infected my spinach plant :(
0	214	Mon May 25 17:26:50 UTC 2009	insects	AntoineTheReaL	wish i could catch every mosquito in the world n burn em slowly.they been bitin the shit outta me 2day.mosquitos are the assholes of insects
0	215	Mon May 25 17:27:05 UTC 2009	insects	jonwolpert	just got back from church, and I totally hate insects.
0	216	Mon May 25 17:28:50 UTC 2009	mcdonalds	jachshore	Just got mcdonalds goddam those eggs make me sick. O yeah Laker up date go lakers. Not much of an update? Well it's true so suck it
4	217	Mon May 25 17:29:39 UTC 2009	mcdonalds	MamiYessi	omgg i ohhdee want mcdonalds damn i wonder if its open lol =]
0	218	Mon May 25 17:31:21 UTC 2009	exam	jvici0us	History exam studying ugh
0	219	Mon May 25 17:31:43 UTC 2009	exam	enriquenieto	I hate revision, it's so boring! I am totally unprepared for my exam tomorrow :( Things are not looking good...
0	220	Mon May 25 17:31:45 UTC 2009	exam	Drummermatt_182	Higher physics exam tommorow, not lookin forward to it much :(
0	222	Mon May 25 17:32:11 UTC 2009	exam	filmcriticbeta	It's a bank holiday, yet I'm only out of work now. Exam season sucks:(
0	223	Mon May 25 17:34:40 UTC 2009	cheney	lvlocal	Cheney and Bush are the real culprits - http://fwix.com/article/939496
0	224	Mon May 25 17:34:51 UTC 2009	cheney	QCWofNC	Life?s a bitch? and so is Dick Cheney. #p2 #bipart #tlot #tcot #hhrs #GOP #DNC http://is.gd/DjyQ
0	225	Mon May 25 17:35:23 UTC 2009	cheney	jepaco	Dick Cheney's dishonest speech about torture, terror, and Obama. -Fred Kaplan Slate. http://is.gd/DiHg
0	226	Mon May 25 17:37:56 UTC 2009	republican	ImportantQuotes	"The Republican party is a bunch of anti-abortion zealots who couldn't draw flies to a dump." -- Neal Boortz (just now, on the radio)
0	227	Mon May 25 17:46:06 UTC 2009	twitter api	fwhamm	is Twitter's connections API broken? Some tweets didn't make it to Twitter...
0	228	Mon May 25 17:46:24 UTC 2009	twitter api	jos897	i srsly hate the stupid twitter API timeout thing, soooo annoying!!!!! :(
4	233	Wed May 27 00:34:21 UTC 2009	jquery book	jystewart	@psychemedia I really liked @kswedberg's "Learning jQuery" book. http://bit.ly/pg0lT is worth a look too
2	234	Wed May 27 00:34:47 UTC 2009	jquery book	cfbloggers	jQuery UI 1.6 Book Review - http://cfbloggers.org/?c=30631
4	235	Wed May 27 00:38:56 UTC 2009	goodby silverste	CaerusMe	Very Interesting Ad from Adobe by Goodby, Silverstein &amp; Partners - YouTube - Adobe CS4: Le Sens Propre http://bit.ly/VprpT
4	236	Wed May 27 00:39:13 UTC 2009	goodby silverste	HallandPartners	Goodby Silverstein agency new site! http://www.goodbysilverstein.com/ Great!
4	237	Wed May 27 00:39:21 UTC 2009	goodby silverste	suedecrush	RT @designplay Goodby, Silverstein's new site: http://www.goodbysilverstein.com/ I enjoy it. *nice find!*
4	238	Wed May 27 00:41:13 UTC 2009	goodby silverste	_imageworks	The ever amazing Psyop and Goodby Silverstein &amp; Partners for HP! http://bit.ly/g2rU8 Have to go play with After Effects now!
4	239	Wed May 27 00:42:22 UTC 2009	wieden	dustinrowley	top ten most watched on Viral-Video Chart.  Love the nike #mostvaluablepuppets campaign from Wieden &amp; Kennedy http://bit.ly/nR1n9
4	251	Wed May 27 23:49:47 UTC 2009	g2	xzela	zomg!!! I have a G2!!!!!!!
4	252	Wed May 27 23:49:59 UTC 2009	g2	mobiledreams	Ok so lots of buzz from IO2009 but how lucky are they - a Free G2!! http://is.gd/Hyzl
4	253	Wed May 27 23:50:46 UTC 2009	g2	crashfaster	just got a free G2 android at google i/o!!!
4	254	Wed May 27 23:51:30 UTC 2009	g2	dragonal	Guess I'll be retiring my G1 and start using my developer G2 woot #googleio
2	255	Wed May 27 23:56:56 UTC 2009	googleio	mastooo	At GWT fireside chat @googleio
4	256	Wed May 27 23:59:18 UTC 2009	googleio	maex242	I am happy for Philip being at GoogleIO today
4	317	Sat May 30 17:46:39 UTC 2009	lakers	specs20	Lakers played great!  Cannot wait for Thursday night Lakers vs. ???
2	328	Sun May 31 06:51:14 UTC 2009	viral marketing	BrandKarma	Hi there, does anyone have a great source for advice on viral marketing?... http://link.gs/YtZ8
4	329	Sun May 31 06:51:30 UTC 2009	viral marketing	mattcad	Judd Apatow creates fake sitcom on NBC.com to market his new movie... viral marketing at its best. http://is.gd/K0yK
2	330	Sun May 31 06:51:34 UTC 2009	viral marketing	grahamgrimshaw	Here's A case study on how to use viral marketing to add over 10,000 people to your list http://snipr.com/i50oz
0	331	Sun May 31 06:51:44 UTC 2009	viral marketing	nicoleisms	VIRAL MARKETING FAIL. This Acia Pills brand oughta get shut down for hacking into people's messenger's.  i get 5-6 msgs in a day! Arrrgh!
4	388	Tue Jun 02 02:54:09 UTC 2009	"night at the mu	Cristinellaa	watching Night at The Museum . Lmao
4	389	Tue Jun 02 02:54:12 UTC 2009	"night at the mu	MzJill	i loved night at the museum!!!
2	390	Tue Jun 02 02:54:22 UTC 2009	"night at the mu	Lynn_Sky	going to see the new night at the museum  movie with my family oh boy a three year old in the movies fuin
4	391	Tue Jun 02 02:54:40 UTC 2009	"night at the mu	bobster56	just got back from the movies.  went to see the new night at the museum with rachel.  it was good
2	392	Tue Jun 02 02:54:44 UTC 2009	"night at the mu	jordanforeman	Just saw the new Night at the Museum movie...it was...okay...lol 7\\10
2	393	Tue Jun 02 02:54:51 UTC 2009	"night at the mu	britree	Going to see night at the museum 2 with tall boy
4	394	Tue Jun 02 02:55:16 UTC 2009	"night at the mu	jellz	@shannyoday I will take you on a date to see night at the museum 2 whenever you want...it looks soooooo good
4	395	Tue Jun 02 02:55:25 UTC 2009	"night at the mu	droherty	no watching The Night At The Museum. Getting Really Good
4	396	Tue Jun 02 02:55:39 UTC 2009	"night at the mu	sarahbrooke	Night at the Museum, Wolverine and junk food - perfect monday!
4	397	Tue Jun 02 02:55:49 UTC 2009	"night at the mu	jeremyempire	saw night at the museum 2 last night.. pretty crazy movie.. but the cast was awesome so it was well worth it. Robin Williams forever!
2	398	Tue Jun 02 02:56:01 UTC 2009	"night at the mu	MirandaClues	I saw Night at the Museum: Battle of the Swithsonian today. It was okay. Your typical [kids] Ben Stiller movie.
2	399	Tue Jun 02 02:56:25 UTC 2009	"night at the mu	RGuad08	Taking Katie to see Night at the Museum.  (she picked it)
0	400	Tue Jun 02 02:56:38 UTC 2009	"night at the mu	xshallsx	Night at the Museum tonite instead of UP. :( oh well. that 4 yr old better enjoy it. LOL
2	401	Tue Jun 02 03:00:25 UTC 2009	gm	EconomyUpdates	GM says expects announcment on sale of Hummer soon - Reuters: WDSUGM says expects announcment on sale of Hummer .. http://bit.ly/4E1Fv
0	402	Tue Jun 02 03:01:10 UTC 2009	gm	mshbrown	It's unfortunate that after the Stimulus plan was put in place twice to help GM on the back of the American people has led to the inevitable
0	403	Tue Jun 02 03:02:17 UTC 2009	gm	misschris62	Tell me again why we are giving more $$ to GM?? We should use that $ for all the programs that support the unemployed.
0	404	Tue Jun 02 03:05:13 UTC 2009	gm	artbynemo	@jdreiss oh yes but if GM dies it will only be worth more boo hahaha
0	405	Tue Jun 02 03:14:36 UTC 2009	time warner	windhamgirl	Time Warner cable is down again 3rd time since Memorial Day bummer!
0	406	Tue Jun 02 03:15:11 UTC 2009	time warner	mmmPi	I would rather pay reasonable yearly taxes for "free" fast internet, than get gouged by Time Warner for a slow connection.
0	407	Tue Jun 02 03:15:23 UTC 2009	time warner	NDEddieMac	NOOOOOOO my DVR just died and I was only half way through the EA presser. Hate you Time Warner
0	408	Tue Jun 02 03:15:54 UTC 2009	time warner	yourboysdot	F*ck Time Warner Cable!!! You f*cking suck balls!!! I have a $700 HD tv &amp; my damn HD channels hardly ever come in. Bullshit!!
0	409	Tue Jun 02 03:16:16 UTC 2009	time warner	Shazzainla	time warner has the worse customer service ever. I will never use them again
0	410	Tue Jun 02 03:16:27 UTC 2009	time warner	dstalk	Time warner is the devil. Worst possible time for the Internet to go out.
0	411	Tue Jun 02 03:16:37 UTC 2009	time warner	ernestalfonso	Fuck no internet damn time warner!
0	412	Tue Jun 02 03:16:49 UTC 2009	time warner	kaaatiee	time warner really picks the worst time to not work. all i want to do is get to mtv.com so i can watch the hills. wtfffff.
0	413	Tue Jun 02 03:17:04 UTC 2009	time warner	JasonNegron	I hate Time Warner! Soooo wish I had Vios. Cant watch the fricken Mets game w/o buffering. I feel like im watching free internet porn.
0	414	Tue Jun 02 03:17:26 UTC 2009	time warner	elphabablue	Ahh...got rid of stupid time warner today &amp; now taking a nap while the roomies cook for me. Pretty good end for a monday :)
0	415	Tue Jun 02 03:17:55 UTC 2009	time warner	gabe_rp	Time Warner's HD line up is crap.
0	416	Tue Jun 02 03:18:50 UTC 2009	time warner	JoeyCircles	is being fucked by time warner cable. didnt know modems could explode. and Susan Boyle sucks too!
2	417	Tue Jun 02 03:22:07 UTC 2009	time warner	ScooPost	Time Warner Cable Pulls the Plug on 'The Girlfriend Experience' - (www.tinyurl.com/m595fk)
0	418	Tue Jun 02 03:23:50 UTC 2009	time warner	adamjleach	Time Warner Cable slogan: Where calling it a day at 2pm Happens.
2	419	Tue Jun 02 03:24:43 UTC 2009	china	HYPHYROCKSTAR	Rocawear Heads to China, Building 300 Stores  - http://tinyurl.com/nofet3
2	420	Tue Jun 02 03:24:47 UTC 2009	china	BreakingBizNews	Climate focus turns to Beijing: The United Nations, the US and European governments have called on China to co-o.. http://tinyurl.com/lto92n
2	421	Tue Jun 02 03:24:57 UTC 2009	china	myfoxdc	myfoxdc Barrie Students Back from Trip to China: A Silver Spring high school's class trip to China has en.. http://tinyurl.com/nlhqba
2	422	Tue Jun 02 03:25:01 UTC 2009	china	drlombardo	Three China aerospace giants develop Tianjin Binhai  New Area,  22.9 B yuan invested   http://bit.ly/mMiDv
2	423	Tue Jun 02 03:27:27 UTC 2009	gm	sinostream	http://xi.gs/04FO GM CEO: China will continue to be key partner
2	424	Tue Jun 02 03:27:48 UTC 2009	gm	hammerauto	RT @LATimesautos is now the time to buy a GM car? http://bit.ly/nRzlu
0	425	Tue Jun 02 03:29:23 UTC 2009	surgery	scoralli	Recovering from surgery..wishing @julesrenner was here :(
4	426	Tue Jun 02 03:29:53 UTC 2009	dentist	sardonnica	My wrist still hurts. I have to get it looked at. I HATE the dr/dentist/scary places. :( Time to watch Eagle eye. If you want to join, txt!
2	427	Tue Jun 02 03:32:14 UTC 2009	dentist	BChasnov	Dentist tomorrow. Have to brush well in the morning. Like I make my hair all nice before I get it cut. Why?
0	428	Tue Jun 02 03:32:33 UTC 2009	dentist	LILJIZZEL	THE DENTIST LIED! " U WON'T FEEL ANY DISCOMORT! PROB WON'T EVEN NEED PAIN PILLS" MAN U TWIPPIN THIS SHIT HURT!! HOW MANY PILLS CAN I TAKE!!
0	429	Tue Jun 02 03:32:45 UTC 2009	dentist	giz2000	@kirstiealley my dentist is great but she's expensive...=(
2	430	Tue Jun 02 03:33:30 UTC 2009	dentist	cmonaussiecmon	@kirstiealley Pet Dentist http://www.funnyville.com/fv/pictures/dogdentures.shtml
4	431	Tue Jun 02 03:33:53 UTC 2009	dentist	sorayabouby	is studing math ;) tomorrow exam and dentist :)
0	432	Tue Jun 02 03:34:27 UTC 2009	dentist	jeffreymodest	my dentist was wrong... WRONG
0	433	Tue Jun 02 03:34:39 UTC 2009	dentist	yowneh	Going to the dentist later.:|
0	434	Tue Jun 02 03:34:58 UTC 2009	dentist	CWilliams_Rltr	Son has me looking at cars online.  I hate car shopping.  Would rather go to the dentist!  Anyone with a good car at a good price to sell?
2	435	Tue Jun 02 04:29:16 UTC 2009	baseball	SimpleManJess	NCAA Baseball Super Regional - Rams Club http://bit.ly/Ro7nx
2	436	Tue Jun 02 04:29:26 UTC 2009	baseball	H3LLGWAR	just started playing Major League Baseball 2K9. http://raptr.com/H3LLGWAR
2	437	Tue Jun 02 04:29:30 UTC 2009	baseball	LouisvilleGrads	Cardinals baseball advance to Super Regionals. Face CS-Fullerton Friday.
2	438	Tue Jun 02 04:39:24 UTC 2009	sony	SwainFamily	Sony coupon code.. Expires soon.. http://www.coupondork.com/r/1796
2	439	Tue Jun 02 04:39:49 UTC 2009	safeway	neeeelia	waiting in line at safeway.
0	440	Tue Jun 02 04:39:59 UTC 2009	safeway	ryanlipert	luke and i got stopped walking out of safeway and asked to empty our pockets and lift our shirts. how jacked up is that?
2	441	Tue Jun 02 04:41:03 UTC 2009	safeway	evan	Did not realize there is a gym above Safeway!
2	442	Tue Jun 02 04:41:07 UTC 2009	safeway	ronjon	@XPhile1908 I have three words for you: "Safeway dot com"
4	443	Tue Jun 02 04:41:19 UTC 2009	safeway	missleigh	Safeway is very rock n roll tonight
2	444	Tue Jun 02 04:41:36 UTC 2009	safeway	SharkKMV	Bout to hit safeway I gotta eat
2	445	Tue Jun 02 04:41:48 UTC 2009	safeway	bosZmom	Jake's going to safeway!
2	446	Tue Jun 02 04:41:58 UTC 2009	safeway	penlynwilson	Found a safeway. Picking up a few staples.
2	447	Tue Jun 02 04:42:28 UTC 2009	safeway	mobileadgirl	Safeway Super-marketing via mobile coupons http://bit.ly/ONH7w
0	448	Tue Jun 02 04:42:34 UTC 2009	safeway	phantomzangel	The safeway bathroom still smells like ass!
0	449	Tue Jun 02 04:42:46 UTC 2009	safeway	BreezB	At safeway on elkhorn, they move like they're dead!
2	451	Tue Jun 02 06:53:03 UTC 2009	eating	nonstopdiets	Your Normal Weight (and How to Get There) ? Normal Eating Blog http://bit.ly/ZeT8O
2	452	Tue Jun 02 06:53:11 UTC 2009	eating	janelleshanks	Is Eating and Watching Movies....
2	456	Tue Jun 02 06:53:27 UTC 2009	eating	hanaho	eating sashimi
2	457	Tue Jun 02 06:53:37 UTC 2009	eating	Fiel	is eating  home made yema
2	458	Tue Jun 02 06:54:16 UTC 2009	eating	aaindefenzo	eating cake
2	2100	Sun May 17 17:30:03 UTC 2009	san francisco	bKnapp	breakers. in San Francisco, CA http://loopt.us/4v88Bw.t
4	531	Thu Jun 04 16:49:55 UTC 2009	nike	CrownRoyal8	i love Dwight Howard's vitamin water commercial... now i wish he was with NIKE and not adidas. lol.
0	532	Thu Jun 04 16:52:36 UTC 2009	nike	MrGQ	Found NOTHING at Nike Factory :/ Off to Banana Republic Outlet! http://myloc.me/2zic
2	533	Thu Jun 04 16:53:09 UTC 2009	nike	ngngfrancis	iPhone May Get Radio Tagging and Nike  : Recently-released iTunes version 8.2 suggests that VoiceOver functional.. http://tinyurl.com/oq5ctc
4	534	Thu Jun 04 16:54:23 UTC 2009	nike	mikelongden	is lovin his Nike  already and that's only from running on the spot in his bedroom
2	556	Sun Jun 07 01:12:50 UTC 2009	jquery	imgsearch	Launched! http://imgsearch.net  #imgsearch #ajax #jquery #webapp
4	557	Sun Jun 07 01:13:52 UTC 2009	jquery	teagone	@matthewcyan I finally got around to using jquery to make my bio collapse. Yay for slide animations.
2	558	Sun Jun 07 01:14:08 UTC 2009	jquery	marcroberts	RT @jquery: The Ultimate jQuery List - http://jquerylist.com/
2	559	Sun Jun 07 01:14:15 UTC 2009	jquery	jacobrothstein	I just extracted and open-sourced a jQuery plugin from Stormweight to highlight text with a regular expression: http://bit.ly/ybJKb
2	560	Sun Jun 07 01:14:19 UTC 2009	jquery	bedroomation	@anna_debenham what was the php jquery hack?
2	561	Sun Jun 07 01:14:42 UTC 2009	jquery	vmkobs	jQuery Cheat Sheet http://www.javascripttoolbox.com/jquery/cheatsheet/
2	562	Sun Jun 07 01:15:04 UTC 2009	jquery	NewTechBooks	Beginning JavaScript and CSS Development with jQuery #javascript #css #jquery http://bit.ly/TO3e5
4	563	Sun Jun 07 03:27:57 UTC 2009	warren buffet	adthomas3	@PDubyaD right!!! LOL we'll get there!! I have high expectations, Warren Buffet style.
4	564	Sun Jun 07 03:28:08 UTC 2009	warren buffet	Alfred04654	RT @blknprecious1: RT GREAT @dbroos "Someone's sitting in the shade today because someone planted a tree a long time ago."- Warren Buffet
2	565	Sun Jun 07 03:28:50 UTC 2009	warren buffet	LovelyMiska	Warren Buffet on the economy http://ping.fm/Lau0p
4	566	Sun Jun 07 03:29:15 UTC 2009	warren buffet	adamgilmer	Warren Buffet became (for a time) the richest man in the United States, not by working but investing in 1 Big idea which lead to the fortune
4	567	Sun Jun 07 17:42:47 UTC 2009	notre dame schoo	PatfaceCatface	According to the create a school, Notre Dame will have 7 receivers in NCAA 10 at 84 or higher rating :) *sweet*
2	568	Sun Jun 07 17:42:56 UTC 2009	notre dame schoo	MY_NBA_PLAYOFFS	All-Star Basketball Classic Tuesday Features Top Talent: Chattanooga's Notre Dame High School will play host.. http://bit.ly/qltJA
4	569	Sun Jun 07 21:38:16 UTC 2009	kindle2	rachaelbender	@BlondeBroad it's definitely under warranty &amp; my experience is the amazon support for kindle is great! had to contact them about my kindle2
2	570	Sun Jun 07 21:38:42 UTC 2009	kindle2	aqrinc	RT Look, Available !Amazon Kindle2 &amp; Kindle DX, Get it Here: http://short.to/87ub The Top Electronic Book Reader Period, free 2 day ship ...
0	571	Sun Jun 07 21:47:01 UTC 2009	time warner	davepurcell	Time Warner Road Runner customer support here absolutely blows. I hate not having other high-speed net options. I'm ready to go nuclear.
0	572	Sun Jun 07 21:47:20 UTC 2009	time warner	MarleyLuv26	Time Warner cable phone reps r dumber than nails!!!!! UGH! Cable was working 10 mins ago now its not WTF!
0	573	Sun Jun 07 21:47:33 UTC 2009	time warner	acomicbookgirl	@siratomofbones we tried but Time Warner wasn't being nice so we recorded today. :)
0	574	Sun Jun 07 21:47:42 UTC 2009	time warner	MichelleEReagan	OMG - time warner f'ed up my internet install - instead of today  its now NEXT saturday - another week w/o internet! &amp;$*ehfa^V9fhg[*# fml.
0	575	Sun Jun 07 21:47:50 UTC 2009	time warner	jpaje11	wth..i have never seen a line this loooong at time warner before, ugh.
0	576	Sun Jun 07 21:47:56 UTC 2009	time warner	JessSlevin	Impatiently awaiting the arrival of the time warner guy. It's way too pretty to be inside all afternoon
2	577	Mon Jun 08 00:01:29 UTC 2009	federer	jworthington	Man accosts Roger Federer during French Open http://ff.im/3HCPT
0	578	Mon Jun 08 01:58:49 UTC 2009	"naive bayes"	R0B3rt2	Naive Bayes using EM for Text Classification. Really Frustrating...
4	585	Mon Jun 08 06:28:01 UTC 2009	stanford	TammyT	We went to Stanford University today. Got a tour. Made me want to go back to college. It's also decided all of our kids will go there.
2	586	Mon Jun 08 06:28:05 UTC 2009	stanford	elliottng	Investigation pending on death of Stanford CS prof / Google mentor Rajeev Motwani http://bit.ly/LwOUR tip @techmeme
2	587	Mon Jun 08 06:28:23 UTC 2009	stanford	Shibuki_kun	I'm going to bed. It was a successful weekend. Stanford, here I come.
0	588	Mon Jun 08 06:55:01 UTC 2009	car warranty cal	painhatelove	@KarrisFoxy If you're being harassed by calls about your car warranty, changing your number won't fix that. They call every number. #d-bags
0	589	Mon Jun 08 06:55:36 UTC 2009	car warranty cal	russdavisdotcom	Just blocked United Blood Services using Google Voice. They call more than those Car Warranty guys.
0	594	Mon Jun 08 19:59:16 UTC 2009	at&t	jamesmakeseyes	#at&amp;t is complete fail.
0	595	Mon Jun 08 19:59:23 UTC 2009	at&t	mikeyil	@broskiii OH SNAP YOU WORK AT AT&amp;T DON'T YOU
0	596	Mon Jun 08 19:59:26 UTC 2009	at&t	nakiasmile	@Mbjthegreat i really dont want AT&amp;T phone service..they suck when it comes to having a signal
0	597	Mon Jun 08 20:00:52 UTC 2009	at&t	wiiskey	I say we just cut out the small talk: AT&amp;T's new slogan: F__k you, give us your money. (Apologies to Bob Geldof.)
0	598	Mon Jun 08 20:01:11 UTC 2009	at&t	brkd	pissed about at&amp;t's mid-contract upgrade price for the iPhone (it's $200 more) I'm not going to pay $499 for something I thought was $299
0	599	Mon Jun 08 20:02:01 UTC 2009	at&t	jesusmbaez	Safari 4 is fast :) Even on my shitty AT&amp;T tethering.
0	600	Mon Jun 08 20:06:29 UTC 2009	at&t	ClaystationX	@ims What is AT&amp;T fucking up?
0	601	Mon Jun 08 20:06:57 UTC 2009	at&t	matthewphewes	@springsingfiend @dvyers @sethdaggett @jlshack AT&amp;T dropped the ball and isn't supporting crap with the new iPhone 3.0... FAIL #att SUCKS!!!
0	602	Mon Jun 08 20:08:26 UTC 2009	at&t	chelseabot	@MMBarnhill yay, glad you got the phone! Still, damn you, AT&amp;T.
2	603	Mon Jun 08 20:46:01 UTC 2009	wave sandbox	girgely	Google Wave Developer Sandbox Account Request http://bit.ly/2NYlc
0	613	Tue Jun 09 01:02:11 UTC 2009	bing	StrategisAdv	Talk is Cheap: Bing that, I?ll stick with Google. http://bit.ly/XC3C8
2	2101	Sun May 17 17:30:25 UTC 2009	san francisco	rayceja	Heading to San Francisco
0	614	Tue Jun 09 03:58:40 UTC 2009	summize	fabulousaura	@defsounds WTF is the point of deleting tweets if they can still be found in summize and searches? Twitter, please fix that. Thanks and bye
2	626	Wed Jun 10 05:24:16 UTC 2009	google	drylight	@mattcutts have google profiles stopped showing up in searches? cant see them anymore
4	627	Wed Jun 10 05:24:26 UTC 2009	google	ManiKarthik	@ArunBasilLal I love Google Translator too ! :D Good day mate !
4	1002	Mon May 11 03:17:54 UTC 2009	kindle2	jhall515	reading on my new Kindle2!
4	1003	Mon May 11 03:18:59 UTC 2009	kindle2	HappyWino	My Kindle2 came and I LOVE it! :)
4	1004	Mon May 11 03:21:50 UTC 2009	kindle2	Teaguem2005	LOVING my new Kindle2.  Named her Kendra in case u were wondering. The "cookbook" is THE tool cuz it tells u all the tricks!  Best gift EVR!
0	1005	Mon May 11 03:22:24 UTC 2009	aig	dailysourcedev	The real AIG scandal / http://bit.ly/b82Px
2	1006	Mon May 11 03:27:20 UTC 2009	twitter	gandhirama	Any twitter to aprs apps yet?
2	1007	Mon May 11 03:27:58 UTC 2009	twitter	delpop	45 Pros You Should Be Following on Twitter - http://is.gd/sMbZ
4	1008	Mon May 11 03:29:01 UTC 2009	obama	lingbellbell	Obama is quite a good comedian! check out his dinner speech on CNN :) very funny jokes.
4	1009	Mon May 11 03:29:38 UTC 2009	obama	motsandco	' Barack Obama shows his funny side " &gt;&gt; http://tr.im/l0gY !! Great speech..
4	1010	Mon May 11 03:29:41 UTC 2009	obama	mickou	I like this guy : ' Barack Obama shows his funny side " &gt;&gt; http://tr.im/l0gY !!
4	1011	Mon May 11 03:32:35 UTC 2009	obama	meggentile	Obama's speech was pretty awesome last night! http://bit.ly/IMXUM
4	1012	Mon May 11 03:33:05 UTC 2009	obama	failness	Reading  "Bill Clinton Fail - Obama Win?" http://tinyurl.com/pcyxj7
4	1013	Mon May 11 03:33:43 UTC 2009	obama	kledy	Obama More Popular Than U.S. Among Arabs: Survey: President Barack Obama's popularity in leading Arab countries .. http://tinyurl.com/prlvqu
4	1014	Mon May 11 03:34:04 UTC 2009	obama	LaurelEdelstein	Obama's got JOKES!! haha just got to watch a bit of his after dinner speech from last night... i'm in love with mr. president ;)
0	1015	Mon May 11 05:19:58 UTC 2009	lebron	chelseabigass	LEbron james got in a car accident i guess..just heard it on evening news...wow i cant believe it..will he be ok ? http://twtad.com/69750
4	1016	Mon May 11 05:20:22 UTC 2009	lebron	Smitty478	is it me or is this the best the playoffs have been in years oh yea lebron and melo in the finals
4	1017	Mon May 11 05:20:41 UTC 2009	lebron	PurpleGorillaZz	@khalid0456 No, Lebron is the best
4	1018	Mon May 11 05:20:48 UTC 2009	lebron	MsStacha	@the_real_usher LeBron is cool.  I like his personality...he has good character.
4	1019	Mon May 11 05:21:25 UTC 2009	lebron	undefeated310	Watching Lebron highlights. Damn that niggas good
4	1020	Mon May 11 05:21:57 UTC 2009	lebron	BI_POLAROID	@Lou911 Lebron is MURDERING shit.
4	1021	Mon May 11 05:22:06 UTC 2009	lebron	IAmNoeAngel	@uscsports21 LeBron is a monsta and he is only 24. SMH The world ain't ready.
4	1022	Mon May 11 05:22:26 UTC 2009	lebron	ksmedia	@cthagod when Lebron is done in the NBA he will probably be greater than Kobe. Like u said Kobe is good but there alot of 'good' players.
4	1023	Mon May 11 05:22:40 UTC 2009	lebron	PrEttIBoIGaRy	KOBE IS GOOD BT LEBRON HAS MY VOTE
0	1024	Mon May 11 05:22:51 UTC 2009	lebron	kellan38	Kobe is the best in the world not lebron .
4	1025	Mon May 11 05:52:44 UTC 2009	world cup	jsincere150	@asherroth World Cup 2010 Access?? Damn, that's a good look!
4	1026	Mon May 11 05:53:33 UTC 2009	world cup 2010	biren	Just bought my tickets for the 2010 FIFA World Cup in South Africa. Its going to be a great summer. http://bit.ly/9GEZI
2	1031	Mon May 11 19:49:06 UTC 2009	fred wilson	DoAndroidsDream	Share: Disruption...Fred Wilson's slides for his talk at Google HQ  http://bit.ly/Bo8PG
0	2010	Thu May 14 02:58:11 UTC 2009	"booz allen"	Moc5085	I have to go to Booz Allen Hamilton for a 2hr meeting :(  But then i get to go home :)
4	2011	Thu May 14 03:41:01 UTC 2009	indian election	citizenofindia	The great Indian tamasha truly will unfold from May 16, the result day for Indian General Election.
4	2012	Thu May 14 05:23:32 UTC 2009	kindle2	Melonze	@crlane I have the Kindle2. I've seen pictures of the DX, but haven't seen it in person. I love my Kindle - I'm on it everyday.
4	2013	Thu May 14 05:24:33 UTC 2009	kindle2	SPReviews	@criticalpath Such an awesome idea - the  continual learning program with a Kindle2  http://bit.ly/1ZLfF
2	2014	Thu May 14 05:25:08 UTC 2009	40d	ha_nobita	ok.. do nothing.. just thinking about 40D
4	2015	Thu May 14 05:26:03 UTC 2009	40d	Jen2Squared	@faithbabywear Ooooh, what model are you getting??? I have the 40D and LOVE LOVE LOVE LOVE it!
4	2018	Fri May 15 06:45:54 UTC 2009	india election	__new	The Times of India: The wonder that is India's election. http://bit.ly/p7u1H
4	2080	Sat May 16 16:18:47 UTC 2009	google	BillVick	http://is.gd/ArUJ Good video from Google on using search options.
0	2083	Sat May 16 22:42:21 UTC 2009	itchy	xmikeflhxcx	@ambcharlesfield lol. Ah my skin is itchy :( damn lawnmowing.
0	2084	Sat May 16 22:42:40 UTC 2009	itchy	Timl9068	itchy back!! dont ya hate it!
4	2085	Sat May 16 23:48:10 UTC 2009	stanford	PassionModel	Stanford Charity Fashion Show a top draw http://cli.gs/NeNuAH
4	2086	Sat May 16 23:48:38 UTC 2009	stanford	TechUpdater	Stanford University?s Facebook Profile is One of the Most Popular Official University Pages - http://tinyurl.com/p5b3fl
4	2087	Sat May 16 23:58:44 UTC 2009	lyx	robotickilldozr	Lyx is cool.
4	2093	Sun May 17 15:04:50 UTC 2009	Danny Gokey	TheDALiSiA	SOOO DISSAPOiNTED THEY SENT DANNY GOKEY HOME... YOU STiLL ROCK ...DANNY ... MY HOMETOWN HERO !! YEAH MiLROCKEE!!
4	2094	Sun May 17 15:04:55 UTC 2009	Danny Gokey	fashion_retweet	RT @PassionModel 'American Idol' fashion: Adam Lambert tones down, Danny Gokey cute ... http://cli.gs/7JWSHV
4	2095	Sun May 17 15:05:07 UTC 2009	Danny Gokey	angelkim17	@dannygokey I love you DANNY GOKEY!! :)
2	2096	Sun May 17 15:10:58 UTC 2009	gm	download11	RT @justindavey: RT @tweetmeme GM OnStar now instantly sends accident location coordinates to 911 | GPS Obsessed http://bit.ly/16szL1
0	2097	Sun May 17 17:27:54 UTC 2009	sleep	nichole17	so tired. i didn't sleep well at all last night.
0	2098	Sun May 17 17:29:05 UTC 2009	san francisco	xstaylor	Boarding plane for San Francisco in 1 hour; 6 hr flight. Blech.
0	2099	Sun May 17 17:29:41 UTC 2009	san francisco	miwahh	bonjour San Francisco. My back hurts from last night..
4	2102	Sun May 17 17:30:48 UTC 2009	san francisco	deepikaC	With my best girl for a few more hours in San francisco. Mmmmmfamily is wonderful!
0	2103	Sun May 17 17:31:44 UTC 2009	aig	kaaslaw	F*** up big, or go home - AIG
4	2104	Sun May 17 17:34:55 UTC 2009	star trek	Pgobb	Went to see the Star Trek movie last night.  Very satisfying.
4	2105	Sun May 17 17:35:23 UTC 2009	star trek	qrboy85	I can't wait, going to see star trek tonight!!
4	2106	Sun May 17 17:35:58 UTC 2009	star trek	adamrisser	Star Trek was as good as everyone said!!
4	2107	Mon May 18 01:13:21 UTC 2009	Malcolm Gladwell	GoodPhoenix	am loving new malcolm gladwell book - outliers
4	2108	Mon May 18 01:13:42 UTC 2009	Malcolm Gladwell	silentcarto	I highly recommend Malcolm Gladwell's 'The Tipping Point.' My next audiobook will probably be one of his as well.
0	2109	Mon May 18 01:14:02 UTC 2009	Malcolm Gladwell	kerrrrrr	Malcolm Gladwell is a genius at tricking people into not realizing he's a fucking idiot
0	2110	Mon May 18 01:14:35 UTC 2009	Malcolm Gladwell	bling_crosby	@sportsguy33 hey no offense but malcolm gladwell is a pretenious, annoying cunt and he brings you down. cant read his shit
4	2111	Mon May 18 01:15:31 UTC 2009	Malcolm Gladwell	ripplemdk	RT @clashmore: http://bit.ly/SOYv7  Great article by Malcolm Gladwell.
4	2112	Mon May 18 01:16:10 UTC 2009	Malcolm Gladwell	drewlew34	I seriously underestimated Malcolm Gladwell.  I want to meet this dude.
0	2113	Mon May 18 01:21:12 UTC 2009	comcast	MrsGinobili	i hate comcast right now. everything is down cable internet &amp; phone....ughh what am i to do
0	2114	Mon May 18 01:22:40 UTC 2009	comcast	stormygirl223	Comcast sucks.
0	2115	Mon May 18 01:23:00 UTC 2009	comcast	theZoctor	The day I never have to deal with Comcast again will rank as one of the best days of my life.
0	2116	Mon May 18 01:23:30 UTC 2009	comcast	motkeps	@Dommm did comcast fail again??
2	2117	Mon May 18 03:10:34 UTC 2009	"twitter api"	YouAnswer	How do you use the twitter API?... http://bit.ly/4VBhH
0	2118	Mon May 18 03:10:52 UTC 2009	"twitter api"	marekdsi	curses the Twitter API limit
0	2119	Mon May 18 03:11:27 UTC 2009	"twitter api"	mikecane	Now I can see why Dave Winer screams about lack of Twitter API, its limitations and access throttles!
2	2120	Mon May 18 03:11:50 UTC 2009	"twitter api"	trivektor	testing Twitter API
0	2121	Mon May 18 03:12:09 UTC 2009	"twitter api"	rdoc420	Arg. Twitter API is making me crazy.
2	2122	Mon May 18 03:12:23 UTC 2009	"twitter api"	Hot_Tweets	Testing Twitter API. Remote Update
4	2137	Tue May 19 16:20:40 UTC 2009	wolfram alpha	jeffswhite	I'm really loving the new search site Wolfram/Alpha. Makes Google seem so ... quaint. http://www72.wolframalpha.com/
0	2138	Tue May 19 16:21:05 UTC 2009	wolfram alpha	marriop	#wolfram Alpha SUCKS! Even for researchers the information provided is less than you can get from #google or #wikipedia, totally useless!
4	2139	Wed May 20 02:37:18 UTC 2009	nike	kewpiezmom	Off to the NIKE factory!!!
4	2140	Wed May 20 02:38:17 UTC 2009	nike	Chet_Lemon	New nike muppet commercials are pretty cute. Why do we live together again?
2	2141	Wed May 20 02:38:28 UTC 2009	nike	sneakerfiles	New blog post: Nike Zoom LeBron Soldier 3 (III) - White / Black - Teal http://bit.ly/rouUS
2	2142	Wed May 20 02:38:47 UTC 2009	nike	nikeblog	New blog post: Nike Trainer 1 http://bit.ly/394bp
0	2143	Wed May 20 02:39:22 UTC 2009	nike	laurenornot	@Fraggle312 oh those are awesome! i so wish they weren't owned by nike :(
4	2158	Sat May 23 04:23:54 UTC 2009	shoreline amphit	julieules	@tonyhawk http://twitpic.com/5c7uj - AWESOME!!! Seeing the show Friday at the Shoreline Amphitheatre. Never seen NIN before. Can't wait. ...
0	2159	Sat May 23 20:43:16 UTC 2009	weka	gux_kung	arhh, It's weka bug. = =" and I spent almost two hours to find that out. crappy me
4	2160	Sun May 24 16:18:46 UTC 2009	50d	GrfxGuru	@mitzs hey bud :) np I do so love my 50D, although I'd love a 5D mkII more
4	2161	Sun May 24 16:18:54 UTC 2009	50d	joelgoodman	@jonduenas @robynlyn just got us a 50D for the office. :D
4	2162	Sun May 24 16:19:00 UTC 2009	50d	trevorcgibson	Just picked up my new Canon 50D...it's beautiful!!  Prepare for some seriously awesome photography!
4	2163	Sun May 24 16:19:03 UTC 2009	50d	ashpeckham	Just got my new toy. Canon 50D. Love love love it!
4	2164	Sun May 24 20:48:13 UTC 2009	lambda calculus	davidivins	Learning about lambda calculus :)
2	2165	Sun May 24 20:49:34 UTC 2009	east palo alto	SCSanFrancisco	#jobs #sittercity Help with taking care of sick child (East Palo Alto, CA) http://tinyurl.com/qwrr2m
4	2166	Sun May 24 20:50:19 UTC 2009	east palo alto	nfarzan	I'm moving to East Palo Alto!
4	2171	Mon May 25 17:14:58 UTC 2009	stanford	tylerlin	@ atebits I just finished watching your Stanford iPhone Class session. I really appreciate it. You Rock!
4	2172	Mon May 25 17:15:05 UTC 2009	stanford	souleaterjh	@jktweet Hi! Just saw your Stanford talk and really liked your advice. Just saying Hi from Singapore (yes the videos do get around)
2	2173	Mon May 25 17:15:18 UTC 2009	stanford	MbaAdmission	#MBA Admissions Tips Stanford GSB Deadlines and Essay Topics 2009-2010 http://tinyurl.com/pet4fd
2	2174	Mon May 25 17:15:35 UTC 2009	stanford	narain	Ethics and nonprofits - http://bit.ly/qsXRp  #stanford #socialentrepreneurship
4	2175	Mon May 25 17:16:52 UTC 2009	lakers	JuanGir	LAKERS tonight let's go!!!!
4	2176	Mon May 25 17:17:10 UTC 2009	lakers	Alexi_G	Will the Lakers kick the Nuggets ass tonight?
0	2177	Mon May 25 17:18:49 UTC 2009	north korea	one_eighteen	Oooooooh... North Korea is in troubleeeee! http://bit.ly/19epAH
0	2178	Mon May 25 17:19:07 UTC 2009	north korea	FOLKTALE09	Wat the heck is North Korea doing!!??!! They just conducted powerful nuclear tests! Follow the link: http://www.msnbc.msn.com/id/30921379
0	2179	Mon May 25 17:19:30 UTC 2009	north korea	Mvsic	Listening to Obama... Friggin North Korea...
0	2180	Mon May 25 17:21:16 UTC 2009	pelosi	CFURNAROS	I just realized we three monkeys in the white Obama.Biden,Pelosi . Sarah Palin 2012
0	2181	Mon May 25 17:21:30 UTC 2009	pelosi	Rachael90210	@foxnews Pelosi should stay in China and never come back.
0	2182	Mon May 25 17:21:35 UTC 2009	pelosi	TylerSchmidt	Nancy Pelosi gave the worst commencement speech I've ever heard. Yes I'm still bitter about this
0	2183	Mon May 25 17:25:36 UTC 2009	insects	KayJay_x	ugh. the amount of times these stupid insects have bitten me. Grr..
4	2184	Mon May 25 17:25:54 UTC 2009	insects	BecCrew	Prettiest insects EVER - Pink Katydids: http://bit.ly/2Upw2p
0	2185	Mon May 25 17:26:30 UTC 2009	insects	euthanasia86	Just got barraged by a horde of insects hungry for my kitchen light. So scary.
4	2187	Mon May 25 17:29:06 UTC 2009	mcdonalds	connlocks	Just had McDonalds for dinner. :D It was goooood. Big Mac Meal. ;)
4	2188	Mon May 25 17:29:11 UTC 2009	mcdonalds	MamiYessi	AHH YES LOL IMA TELL MY HUBBY TO GO GET ME SUM MCDONALDS =]
4	2190	Mon May 25 17:29:46 UTC 2009	mcdonalds	Yuleineeee	Stopped to have lunch at McDonalds. Chicken Nuggetssss! :) yummmmmy.
4	2191	Mon May 25 17:29:51 UTC 2009	mcdonalds	XrachulX	Could go for a lot of McDonalds. i mean A LOT.
4	2193	Mon May 25 17:31:52 UTC 2009	exam	xKimmelie	my exam went good. @HelloLeonie: your prayers worked (:
4	2194	Mon May 25 17:31:58 UTC 2009	exam	laulaulauren	Only one exam left, and i am so happy for it :D
0	2195	Mon May 25 17:32:22 UTC 2009	exam	elllllen	Math review. Im going to fail the exam.
0	2196	Mon May 25 17:35:08 UTC 2009	cheney	LPSsports43	Colin Powell rocked yesterday on CBS. Cheney needs to shut the hell up and go home.Powell is a man of Honor and served our country proudly
0	2197	Mon May 25 17:35:43 UTC 2009	cheney	joahs	obviously not siding with Cheney here: http://bit.ly/19j2d
4	2202	Tue May 26 22:39:46 UTC 2009	mashable	paulobsf	Absolutely hilarious!!! from @mashable:  http://bit.ly/bccWt
4	2203	Tue May 26 22:40:41 UTC 2009	mashable	christinerose	@mashable I never did thank you for including me in your Top 100 Twitter Authors! You Rock! (&amp; I New Wave :-D) http://bit.ly/EOrFV
2	2204	Wed May 27 00:34:45 UTC 2009	jquery book	cfbloggers	Learning jQuery 1.3 Book Review - http://cfbloggers.org/?c=30629
4	2205	Wed May 27 00:37:30 UTC 2009	jquery book	pdelsignore	RT @shrop: Awesome JQuery reference book for Coda! http://www.macpeeps.com/coda/ #webdesign
4	2206	Wed May 27 00:38:44 UTC 2009	goodby silverste	bskatz	I've been sending e-mails like crazy today to my contacts...does anyone have a contact at Goodby SIlverstein...I'd love to speak to them
2	2207	Wed May 27 00:39:05 UTC 2009	goodby silverste	sc0ttman	Adobe CS4 commercial by Goodby Silverstein: http://bit.ly/1aikhF
4	2208	Wed May 27 00:39:17 UTC 2009	goodby silverste	designplay	Goodby, Silverstein's new site... http://www.goodbysilverstein.com/ I enjoy it.
4	2220	Wed May 27 23:49:35 UTC 2009	g2	Drisgill	Wow everyone at the Google I/O conference got free G2's with a month of unlimited service
4	2221	Wed May 27 23:50:26 UTC 2009	g2	DearJellyHiJely	@vkerkez dood I got a free google android phone at the I/O conference. The G2!
4	2222	Wed May 27 23:50:48 UTC 2009	g2	itamarw	@Orli the G2 is amazing btw, a HUGE improvement over the G1
4	2223	Wed May 27 23:56:46 UTC 2009	googleio	daynah	HTML 5 Demos! Lots of great stuff to come! Yes, I'm excited. :) http://htmlfive.appspot.com #io2009 #googleio
4	2224	Wed May 27 23:56:53 UTC 2009	googleio	jackdaniels08	@googleio http://twitpic.com/62shi - Yay! Happy place! Place place!  I love Google!
4	2225	Wed May 27 23:57:02 UTC 2009	googleio	alpheus1	#GoogleIO | O3D - Bringing 3d graphics to the browser. Very nice tbh. Funfun.
4	2295	Sun May 31 06:51:32 UTC 2009	viral marketing	joshgammon	Awesome viral marketing for "Funny People" http://www.nbc.com/yo-teach/
2	2296	Sun May 31 06:54:34 UTC 2009	hitler	KatyWheatley1	Watching a programme about the life of Hitler, its only enhancing my geekiness of history.
0	2355	Tue Jun 02 02:54:08 UTC 2009	"night at the mu	TiffanyBakker	saw night at the museum out of sheer desperation. who is funding these movies?
4	2356	Tue Jun 02 02:54:11 UTC 2009	"night at the mu	kezzumz	Night At The Museum 2? Pretty furkin good.
4	2357	Tue Jun 02 02:54:14 UTC 2009	"night at the mu	bodhibuggy	Watching Night at the Museum - giggling.
2	2358	Tue Jun 02 02:54:29 UTC 2009	"night at the mu	HistoryLuV3R	@pambeeslyjenna Jenna, I went to see Night At The Museum 2 today and I was so surprised to see three cast members from The Office...
2	2359	Tue Jun 02 02:54:42 UTC 2009	"night at the mu	ErcDrso	About to watch Night at the Museum with Ryan and Stacy
2	2360	Tue Jun 02 02:54:49 UTC 2009	"night at the mu	aykataoka	Getting ready to go watch Night at the Museum 2.  Dum dum, you give me gum gum!
0	2361	Tue Jun 02 02:55:06 UTC 2009	"night at the mu	Bcapote	Back from seeing 'Star Trek' and 'Night at the Museum.' 'Star Trek' was amazing, but 'Night at the Museum' was; eh.
4	2362	Tue Jun 02 02:55:31 UTC 2009	"night at the mu	crikket_churps	just watched night at the museum 2! so stinkin cute!
4	2363	Tue Jun 02 02:55:50 UTC 2009	"night at the mu	tish_tish	So, Night at the Museum 2 was AWESOME! Much better than part 1. Next weekend we'll see Up.
2	2364	Tue Jun 02 02:56:13 UTC 2009	"night at the mu	Lilimich	I think I may have a new favorite restaurant. On our way to see "Night at the Museum 2".
2	2365	Tue Jun 02 02:56:31 UTC 2009	"night at the mu	stealing_second	UP! was sold out, so i'm seeing Night At The Museum 2. I'm __ years old.
4	2366	Tue Jun 02 02:56:44 UTC 2009	"night at the mu	Pas_de_Cheval	saw the new Night at the Museum and i loved it. Next is to go see UP in 3D
0	2367	Tue Jun 02 03:00:28 UTC 2009	gm	lloydnelson	It is a shame about GM. What if they are forced to make only cars the White House THINKS will sell? What do you think?
0	2368	Tue Jun 02 03:00:40 UTC 2009	gm	overthrow	As u may have noticed, not too happy about the GM situation, nor AIG, Lehman, et al
2	2369	Tue Jun 02 03:00:51 UTC 2009	gm	economywire	Obama: Nationalization of GM to be short-term   (AP) http://tinyurl.com/md347r
0	2370	Tue Jun 02 03:02:00 UTC 2009	gm	tradecruz	@Pittstock $GM good riddance.  sad though.
0	2371	Tue Jun 02 03:02:34 UTC 2009	gm	ZendoDeb	I Will NEVER Buy a Government Motors Vehicle: Until just recently, I drove GM cars. Since 1988, when I bought a .. http://tinyurl.com/lulsw8
0	2372	Tue Jun 02 03:04:22 UTC 2009	gm	rbmshow	Having the old Coca-Cola guy on the GM board is stupid has heck! #tcot #ala
0	2373	Tue Jun 02 03:04:35 UTC 2009	gm	Ash_Craigslist	#RantsAndRaves The worst thing about GM (concord / pleasant hill / martinez): is the fucking UAW. ..   http://buzzup.com/4ueb
0	2374	Tue Jun 02 03:04:45 UTC 2009	gm	BarackProblema	Give a man a fish, u feed him for the day. Teach him to fish, u feed him for life. Buy him GM, and u F**K him over for good.
0	2375	Tue Jun 02 03:05:18 UTC 2009	gm	ericdano	The more I hear about this GM thing the more angry I get. Billions wasted, more bullshit. All for something like 40k employees and all the..
0	2376	Tue Jun 02 03:05:50 UTC 2009	gm	tradecruz	@QuantTrader i own a GM car and it is junk as far as quality compared to a honda
0	2377	Tue Jun 02 03:06:40 UTC 2009	gm	ktsophie	sad day...bankrupt GM
0	2378	Tue Jun 02 03:06:45 UTC 2009	gm	nicolemg415	is upset about the whole GM thing. life as i know it is so screwed up
0	2379	Tue Jun 02 03:14:21 UTC 2009	time warner	ProofofVenom	whoever is running time warner needs to be repeatedly raped by a rhino so they understand the consequences of putting out shitty cable svcs
2	2380	Tue Jun 02 03:14:32 UTC 2009	time warner	flywire	Time Warner CEO hints at online fees for magazines      (AP) - Read from Mountain View,United States. Views 16209 http://bit.ly/UdFCH
0	2381	Tue Jun 02 03:15:18 UTC 2009	time warner	batchoutlost	#WFTB Joining a bit late. My connection was down (boo time warner)
0	2382	Tue Jun 02 03:15:32 UTC 2009	time warner	floatingatoll	Cox or Time Warner?  Cox is cheaper and gets a B on dslreports.  TW is more expensive and gets a C.
0	2383	Tue Jun 02 03:15:52 UTC 2009	time warner	kippy2	i am furious with time warner and their phone promotions!
0	2384	Tue Jun 02 03:16:23 UTC 2009	time warner	iamtony	Just got home from chick-fil-a with the boys. Damn my internets down =( stupid time warner
0	2385	Tue Jun 02 03:16:42 UTC 2009	time warner	dcwhip	could time-warner cable suck more?  NO.
0	2386	Tue Jun 02 03:17:03 UTC 2009	time warner	JustGLB	Pissed at Time Warner for causin me to have slow internet problems
0	2387	Tue Jun 02 03:18:04 UTC 2009	time warner	chaneykyoto	@sportsguy33 Ummm, having some Time Warner problems?
0	2388	Tue Jun 02 03:18:49 UTC 2009	time warner	ldmullen	You guys see this?  Why does Time Warner have to suck so much ass?  Really wish I could get U-Verse at my apartment. http://bit.ly/s594j
0	2389	Tue Jun 02 03:22:08 UTC 2009	time warner	JohnTurlington	RT @sportsguy33 The upside to Time Warner: unhelpful phone operators   superslow on-site service. Crap, that's not an upside.
0	2390	Tue Jun 02 03:22:19 UTC 2009	time warner	johnscleary	RT @sportsguy33: New Time Warner slogan: "Time Warner, where we make you long for the days before cable."
0	2391	Tue Jun 02 03:23:36 UTC 2009	time warner	ecormany	confirmed: it's Time Warner's fault, not Facebook's, that fb is taking about 3 minutes to load. so tempted to switch to verizon =/
0	2392	Tue Jun 02 03:24:00 UTC 2009	time warner	matsonj	@sportsguy33 Time Warner = epic fail
2	2393	Tue Jun 02 03:24:45 UTC 2009	china	YildirimNews	Lawson to head Newedge Hong Kong http://bit.ly/xLQSD #business #china
2	2394	Tue Jun 02 03:24:59 UTC 2009	china	zedomax	Weird Piano Guitar House in China! http://u2s.me/72i8
2	2395	Tue Jun 02 03:27:19 UTC 2009	gm	theprovince	Send us your GM/Chevy photos http://tinyurl.com/luzkpq
0	2396	Tue Jun 02 03:27:34 UTC 2009	gm	PeteHall	I know. How sad is that?  RT @caseymercier: 1st day of hurricane season. That's less scarey than govt taking over GM.
0	2397	Tue Jun 02 03:27:52 UTC 2009	gm	ram_zone	GM files Bankruptcy, not a good sign...
4	2398	Tue Jun 02 03:28:41 UTC 2009	yankees	irishyanks	yankees won mets lost. its a good day.
4	2399	Tue Jun 02 03:29:47 UTC 2009	dentist	Erinthebigsis	My dentist appt today was actually quite enjoyable.
0	2400	Tue Jun 02 03:32:10 UTC 2009	dentist	cassieeeelove	I hate the effing dentist.
2	2401	Tue Jun 02 03:32:28 UTC 2009	dentist	johnnyt183	@stevemoakler i had a dentist appt this morning and had the same conversation!
0	2402	Tue Jun 02 03:33:25 UTC 2009	dentist	b_bassi	@kirstiealley I hate going to the dentist.. !!!
0	2403	Tue Jun 02 03:34:23 UTC 2009	dentist	cmg11	i hate the dentist....who invented them anyways?
0	2404	Tue Jun 02 03:34:37 UTC 2009	dentist	arianaflyy	this dentist's office is cold :/
2	2405	Tue Jun 02 03:34:50 UTC 2009	dentist	jettellis	Check this video out -- David After Dentist http://bit.ly/47aW2
2	2406	Tue Jun 02 03:35:05 UTC 2009	dentist	AdiOpERsOcoM	First dentist appointment [in years] on Wednesday possibly.
2	2407	Tue Jun 02 04:29:14 UTC 2009	baseball	HallofChampions	Tom Shanahan's latest column on SDSU and its NCAA Baseball Regional appearance: http://ow.ly/axhu
2	2408	Tue Jun 02 04:29:17 UTC 2009	baseball	SimpleManJess	BaseballAmerica.com: Blog: Baseball America Prospects Blog ? Blog ... http://bit.ly/EtT8a
2	2409	Tue Jun 02 04:29:28 UTC 2009	baseball	Oregon_Live	Portland city politics may undo baseball park http://tinyurl.com/lpjquj
2	2410	Tue Jun 02 04:39:41 UTC 2009	safeway	AirDye	RT @WaterSISWEB: CA Merced's water bottled by Safeway, resold at a profit: Wells are drying up across the county http://tinyurl.com/mb573s
2	2411	Tue Jun 02 04:40:09 UTC 2009	safeway	csquaredx	dropped her broccoli walking home from safeway! ;( so depressed
2	2412	Tue Jun 02 04:41:05 UTC 2009	safeway	XPhile1908	@ronjon we don't have Safeway.
4	2413	Tue Jun 02 04:41:13 UTC 2009	safeway	LaurenAmor	Just applied at Safeway!(: Yeeeee!
0	2414	Tue Jun 02 04:41:24 UTC 2009	safeway	TravisJensenSF	@ Safeway. Place is a nightmare right now. Bumming.
2	2415	Tue Jun 02 04:41:32 UTC 2009	safeway	Lovely_Lauren	at safeway with dad
0	2416	Tue Jun 02 04:41:41 UTC 2009	safeway	fugface85	HATE safeway select green tea icecream! bought two cartons, what a waste of money.  &gt;_&lt;
2	2417	Tue Jun 02 04:42:00 UTC 2009	safeway	_JessicaJOY	Safeway with Marvin, Janelle, and Auntie Lhu
2	2418	Tue Jun 02 04:42:29 UTC 2009	safeway	mobileadgirl	Safeway offering mobile coupons http://bit.ly/ONH7w
2	2419	Tue Jun 02 06:52:26 UTC 2009	driving	m_marchesi	Phillies Driving in the Cadillac with the Top Down in Cali, Win 5-3 - http://tinyurl.com/nzcjqa
2	2421	Tue Jun 02 06:53:09 UTC 2009	eating	TheDebtress	Saved money by opting for grocery store trip and stocking food in hotel room fridge vs. eating out every night while out of town.
2	2422	Tue Jun 02 06:53:15 UTC 2009	eating	face_of_boe	Lounging around, eating Taco Bell and watching NCIS before work tonight. Need help staying awake.
2	2426	Tue Jun 02 06:53:30 UTC 2009	eating	LaraAvni	eating breakfast and then school
2	2427	Tue Jun 02 06:53:32 UTC 2009	eating	elpasira	still hungry after eating....
2	2428	Tue Jun 02 06:53:45 UTC 2009	eating	Fitness_101	10 tips for healthy eating ? ResultsBy Fitness Blog :: Fitness ... http://bit.ly/62gFn
2	2429	Tue Jun 02 06:54:11 UTC 2009	eating	ohmurder	with the boyfriend, eating a quesadilla
2	2430	Tue Jun 02 06:54:19 UTC 2009	eating	whoaoblivious	Eating dinner. Meat, chips, and risotto.
2	2502	Thu Jun 04 16:53:02 UTC 2009	nike	stefanjos	got a new pair of nike shoes. pics up later
2	2503	Thu Jun 04 16:53:15 UTC 2009	nike	nsborg	Nike SB Blazer High "ACG" Custom - Brad Douglas - http://timesurl.at/45a448
4	2504	Thu Jun 04 16:53:20 UTC 2009	nike	marieforleo	Nike rocks. I'm super grateful for what I've done with them :) &amp; the European Division of NIKE is BEYOND! @whitSTYLES @muchasmuertes
2	2505	Thu Jun 04 16:53:48 UTC 2009	nike	imflashy	Nike Air Yeezy Khaki/Pink Colorway Release - http://shar.es/bjfN
4	2506	Thu Jun 04 16:54:17 UTC 2009	nike	TheGlossMagazin	@evelynbyrne have you tried Nike  ? V. addictive.
2	2507	Thu Jun 04 16:54:32 UTC 2009	nike	MiniGreek	@erickoston That looks an awful lot like one of Nike's private jets....I'm just sayin....
4	2508	Thu Jun 04 16:54:36 UTC 2009	nike	TracySuter	The Nike Training Club (beta) iPhone app looks very interesting.
0	2529	Sun Jun 07 01:12:38 UTC 2009	jquery	TobyJuanKenobi	argghhhh why won't  my jquery appear in safari bad safari !!!
2	2530	Sun Jun 07 01:12:44 UTC 2009	jquery	inBlogs	DevSnippets : jQuery Tools - Javascript UI Components for the Web... http://inblogs.org/go/hfuqt
2	2531	Sun Jun 07 01:12:46 UTC 2009	jquery	vmkobs	all about Ajax,jquery ,css ,JavaScript and more... (many examples) http://ajaxian.com/
4	2532	Sun Jun 07 01:13:23 UTC 2009	jquery	ghurson	I'm ready to drop the pretenses, I am forever in love with jQuery, and I want to marry it. Sorry ladies, this nerd is jquery.spokenFor.js
2	2533	Sun Jun 07 01:13:34 UTC 2009	jquery	5x1llz	This is cold.. I was looking at google's chart//visualization API and found this jQuery "wrapper" for the API...  http://tinyurl.com/mq52bq
2	2534	Sun Jun 07 01:13:43 UTC 2009	jquery	bryall	I spent most of my day reading a jQuery book. Now to start drinking some delirium tremens.
2	2535	Sun Jun 07 01:14:41 UTC 2009	jquery	vmkobs	jquery Selectors http://codylindley.com/jqueryselectors/
2	2536	Sun Jun 07 01:15:08 UTC 2009	jquery	maheshcha	How to implement a news ticker with jQuery and ten lines of code http://bit.ly/CZnFJ
2	2537	Sun Jun 07 03:27:52 UTC 2009	warren buffet	MyBenchmarkMtg	What's Buffet Doing? Warren Buffett Kicks Butt In Battle of the Boots: Posted By:Alex Crippe.. http://bit.ly/AUIzO
4	2538	Sun Jun 07 03:28:18 UTC 2009	warren buffet	PragCapitalist	SUPER INVESTORS: A great weekend read here from Warren Buffet. Oldie, but a goodie. http://tinyurl.com/oqxgga
2	2539	Sun Jun 07 03:28:26 UTC 2009	warren buffet	SmartMouthBroad	I'm truly braindead.  I couldn't come up with Warren Buffet's name to save my soul
4	2540	Sun Jun 07 03:29:04 UTC 2009	warren buffet	goncalol	reading Michael Palin book, The Python Years...great book. I also recommend Warren Buffet &amp; Nelson Mandela's bio
4	2541	Sun Jun 07 17:42:50 UTC 2009	notre dame schoo	BobtheRobot	I mean, I'm down with Notre Dame if I have to.  It's a good school, I'd be closer to Dan, I'd enjoy it.
0	2543	Sun Jun 07 21:47:45 UTC 2009	time warner	Hittman	I can't watch TV without a Tivo.  And after all these years, the Time/Warner DVR  STILL sucks. http://www.davehitt.com/march03/twdvr.html
4	2544	Mon Jun 08 00:01:27 UTC 2009	federer	aiban	I'd say some sports writers are idiots for saying Roger Federer is one of the best ever in Tennis.  Roger Federer is THE best ever in Tennis
0	2545	Mon Jun 08 00:12:16 UTC 2009	kindle2	nyctimes	I still love my Kindle2 but reading The New York Times on it does not feel natural. I miss the Bloomingdale ads.
4	2546	Mon Jun 08 00:13:48 UTC 2009	kindle2	k8tb52	I love my Kindle2. No more stacks of books to trip over on the way to the loo.
0	2558	Mon Jun 08 19:59:10 UTC 2009	at&t	taylorcarrigan	Although today's keynote rocked, for every great announcement, AT&amp;T shit on us just a little bit more.
0	2559	Mon Jun 08 19:59:50 UTC 2009	at&t	susanjane	@sheridanmarfil - its not so much my obsession with cell phones, but the iphone!  i'm a slave to at&amp;t forever because of it. :)
2	2560	Mon Jun 08 20:01:17 UTC 2009	at&t	stick08	@freitasm oh I see. I thought AT&amp;T were 900MHz WCDMA?
2	2561	Mon Jun 08 20:01:55 UTC 2009	at&t	CaptainCraigos	@Plip Where did you read about tethering support Phil?  Just AT&amp;T or will O2 be joining in?
0	2562	Mon Jun 08 20:06:40 UTC 2009	at&t	iPhoneFuzzball	Fuzzball is more fun than AT&amp;T ;P http://fuzz-ball.com/twitter
0	2563	Mon Jun 08 20:07:44 UTC 2009	at&t	itsolivia	Today is a good day to dislike AT&amp;T. Vote out of office indeed, @danielpunkass
4	2564	Mon Jun 08 20:46:18 UTC 2009	wave sandbox	kyrabeckf	GOT MY WAVE SANDBOX INVITE! Extra excited! Too bad I have class now... but I'll play with it soon enough! #io2009 #wave
0	2575	Tue Jun 09 03:58:11 UTC 2009	summize	auxesis	looks like summize has gone down. too many tweets from WWDC perhaps?
2	2576	Tue Jun 09 05:14:20 UTC 2009	kindle2	deedeewarren	I hope the girl at work  buys my Kindle2
2	2577	Tue Jun 09 05:14:32 UTC 2009	kindle2	RogerSPress	Missed this insight-filled May column: One smart guy looking closely at why he's impressed with Kindle2 http://bit.ly/i0peY @wroush
4	2578	Tue Jun 09 05:17:27 UTC 2009	kindle2	kmozena	@sklososky Thanks so much!!! ...from one of your *very* happy Kindle2 winners ; ) I was so surprised, fabulous. Thank you! Best, Kathleen
0	2579	Tue Jun 09 05:53:40 UTC 2009	iphone app	Jesssssii	Man I kinda dislike Apple right now. Case in point: the iPhone 3GS. Wish there was a video recorder app. Please?? http://bit.ly/DZm1T
4	7011	Wed Jun 10 04:03:37 UTC 2009	kindle2	jimhong	@cwong08 I have a Kindle2 (&amp; Sony PRS-500). Like it! Physical device feels good. Font is nice. Pg turns are snappy enuf. UI a little klunky.
4	7012	Wed Jun 10 04:03:53 UTC 2009	kindle2	Ant_Ward	The #Kindle2 seems the best eReader, but will it work in the UK and where can I get one?
4	7015	Wed Jun 10 05:24:40 UTC 2009	google	popitlockit	I have a google addiction. Thank you for pointing that out, @annamartin123. Hahaha.
2	7055	Wed Jun 10 15:31:56 UTC 2009	visa card	FionaSarah	@ruby_gem My primary debit card is Visa Electron.
2	8051	Wed Jun 10 15:31:59 UTC 2009	visa card	MalloryRayne	Off to the bank to get my new visa platinum card
0	8052	Wed Jun 10 15:32:06 UTC 2009	visa card	_abi_	dearest @google, you rich bastards! the VISA card you sent me doesn't work. why screw a little guy like me?
2	13051	Sat Jun 13 16:23:31 UTC 2009	Bobby Flay	sfkerropi6	has a date with bobby flay and gut fieri from food network
4	13052	Sat Jun 13 16:24:08 UTC 2009	Bobby Flay	tessalau	Excited about seeing Bobby Flay and Guy Fieri tomorrow at the Great American Food &amp; Music Fest!
4	13053	Sat Jun 13 16:24:12 UTC 2009	Bobby Flay	ZFilth	Gonna go see Bobby Flay 2moro at Shoreline. Eat and drink. Gonna be good.
4	13054	Sat Jun 13 16:24:25 UTC 2009	Bobby Flay	annieblane	can't wait for the great american food and music festival at shoreline tomorrow.  mmm...katz pastrami and bobby flay. yes please.
4	13055	Sat Jun 13 16:24:34 UTC 2009	Bobby Flay	LAURAcBRYAN	My dad was in NY for a day, we ate at MESA grill last night and met Bobby Flay. So much fun, except I completely lost my voice today.
0	13073	Sun Jun 14 04:35:33 UTC 2009	latex	NathanChalmers	Fighting with LaTex. Again...
0	13074	Sun Jun 14 04:35:53 UTC 2009	latex	LoonyLongbottom	@Iheartseverus we love you too and don't want you to die!!!!!!  Latex = the devil
0	13075	Sun Jun 14 04:36:07 UTC 2009	latex	QuadError	7 hours. 7 hours of inkscape crashing, normally solid as a rock. 7 hours of LaTeX complaining at the slightest thing. I can't take any more.
2	13076	Sun Jun 14 21:35:58 UTC 2009	iran	musicmuse	How to Track Iran with Social Media: http://bit.ly/2BoqU
0	13077	Sun Jun 14 21:36:04 UTC 2009	iran	sketoaks	Shit's hitting the fan in Iran...craziness indeed #iranelection
0	13078	Sun Jun 14 21:36:09 UTC 2009	iran	jamespenycate	Monday already. Iran may implode. Kitchen is a disaster. @annagoss seems happy. @sebulous had a nice weekend and @goldpanda is great. whoop.
2	14045	Sat Jun 13 16:13:59 UTC 2009	aapl	boardcentral	Twitter Stock buzz: $AAPL $ES_F $SPY $SPX $PALM  (updated: 12:00 PM)
4	14046	Sat Jun 13 16:23:41 UTC 2009	Bobby Flay	JimFacey	getting ready to test out some burger receipes this weekend. Bobby Flay has some great receipes to try. Thanks Bobby.
2	14049	Sat Jun 13 16:24:03 UTC 2009	Bobby Flay	wetfishdesigns	@johncmayer is Bobby Flay joining you?
4	14050	Sat Jun 13 16:24:15 UTC 2009	Bobby Flay	A_TALL_BLONDE	i lam so in love with Bobby Flay... he is my favorite. RT @terrysimpson: @bflay you need a place in Phoenix. We have great peppers here!
0	14069	Sun Jun 14 04:31:12 UTC 2009	latex	rooney_tunes	I just created my first LaTeX file from scratch. That didn't work out very well. (See @amandabittner , it's a great time waster)
4	14070	Sun Jun 14 04:31:23 UTC 2009	latex	roguemovement	using Linux and loving it - so much nicer than windows... Looking forward to using the wysiwyg latex editor!
4	14071	Sun Jun 14 04:31:28 UTC 2009	latex	yomcat	After using LaTeX a lot, any other typeset mathematics just looks hideous.
2	14072	Sun Jun 14 04:31:43 UTC 2009	latex	proggit	Ask Programming: LaTeX or InDesign?: submitted by calcio1 [link] [1 comment] http://tinyurl.com/myfmf7
0	14073	Sun Jun 14 04:32:17 UTC 2009	latex	sam33r	On that note, I hate Word. I hate Pages. I hate LaTeX. There, I said it. I hate LaTeX. All you TEXN3RDS can come kill me now.
4	14074	Sun Jun 14 04:36:34 UTC 2009	latex	iamtheonlyjosie	Ahhh... back in a *real* text editing environment. I &lt;3 LaTeX.
0	14075	Sun Jun 14 21:36:07 UTC 2009	iran	plutopup7	Trouble in Iran, I see. Hmm. Iran. Iran so far away. #flockofseagullsweregeopoliticallycorrect
0	14076	Sun Jun 14 21:36:17 UTC 2009	iran	captain_pete	Reading the tweets coming out of Iran... The whole thing is terrifying and incredibly sad...
\.


--
-- Name: test_table test_table_pk; Type: CONSTRAINT; Schema: public; Owner: maindb_ml
--

ALTER TABLE ONLY public.test_table
    ADD CONSTRAINT test_table_pk PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

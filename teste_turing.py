import streamlit as st
from streamlit.script_runner import RerunException
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from scipy import ndimage
import matplotlib.pyplot as plt

class estagio_II(nn.Module):
  def __init__(self, batch_size=16):
    self.batch_size=batch_size

    super(estagio_II, self).__init__()
    
    self.BA17=nn.Conv2d(1, 1, 5)
    
    self.BA19_1=nn.Linear(672, 112)

    self.BA19_2=nn.Linear(112, 28)
    
    self.BA39_1=nn.Linear(28, 32)

    self.BA39_2=nn.Linear(32, 64)
    
    self.BA39_saida=nn.Linear(64, 10)

    self.relu=nn.ReLU()

  def forward(self, entrada_visual, BA19_antigo):
    concat=torch.cat([entrada_visual, BA19_antigo])
    
    bf_BA17=self.BA17(concat.reshape(1,1,32,28))
    af_BA17=self.relu(bf_BA17)
    res=af_BA17.flatten()
    
    bf_BA19_1=self.BA19_1(res)
    af_BA19_1=self.relu(bf_BA19_1)

    BA19_novo=torch.reshape(af_BA19_1, [4,28])

    bf_BA19_2=self.BA19_2(af_BA19_1)
    af_BA19_2=self.relu(bf_BA19_2)
    
    bf_BA39_1=self.BA39_1(af_BA19_2)
    af_BA39_1=self.relu(bf_BA39_1)

    bf_BA39_2=self.BA39_2(af_BA39_1)
    af_BA39_2=self.relu(bf_BA39_2)

    saida=self.BA39_saida(af_BA39_2)
    
    return torch.reshape(saida, [10]), BA19_novo

  def init_BA19(self):
    tensor=nn.init.kaiming_uniform_(torch.empty([4, 28]))
    return tensor

def gera_imagens():
	dados=pd.read_pickle("./dados_teste_reduzido.pkl")
	n=len(dados)
	inds=np.random.random_integers(0,n-1, 12)
	imagens=[]
	for ind in inds:
		imagens.append((torch.Tensor(dados["imagens"][ind]),
						torch.Tensor(dados["target"][ind]).type(torch.long)))
	del dados, inds
	return imagens

def classifica_imagens(imagens):
	rede = torch.load("./estagio_II")
	BA19_antigo=rede.init_BA19()
	saidas=[]
	for entrada, classe in imagens:
		saida, BA19_antigo=rede(entrada, BA19_antigo)
		saidas.append(int(torch.argmax(saida)))
	del rede, BA19_antigo, entrada, classe
	return saidas

NOMES=["Miguel", "Arthur", "Heitor", "Theo", "Davi", "Helena", "Alice", "Laura", "Valentina", "Heloísa"]
SOBRENOMES=["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes"]

if 'estado' not in st.session_state:
    st.session_state.estado = 'inicio'

st.title('Teste de Turing')

if st.session_state.estado=='inicio':
	st.subheader("Como funciona o Teste?")
	st.markdown("Este é um jogo para dois jogadores que implementa um Teste de Turing simples.")
	st.markdown("Primeiro um dos jogadores, que poderá estar competindo com uma máquina e/ou com outra pessas,"+
							" deverá olhar 12 imagens e identificar qual número está escrito.")
	st.markdown("Em seguida, o segundo jogador, que será o  aplicador do teste, verá as imagens e as classificações dadas pelos"+
							" jogadores (que podem ser até 4 distintos) e deverá tentar identificar se existe uma máquina entre eles.")
	comeca=st.button("Começar o teste")

	tt=st.sidebar.button("O que é o teste de Turing?")
	modelo=st.sidebar.button("Que modelo é esse?")
	if comeca or tt or modelo:
		if comeca:
			st.session_state.estado='identifica'
			st.session_state.imagens=gera_imagens()
		elif tt:
			st.session_state.estado="teste"
		elif modelo:
			st.session_state.estado="modelo"
		st.experimental_rerun()

if st.session_state.estado=='teste':
	st.subheader("O que é o Teste de Turing?")
	st.markdown("O Teste de Turing é como ficou conhecido o teste introduzido por Alan Turing,"+
				" matemático inglês considerado o pai da computação, em seu paper 'Computing Machinery and Intelligence'"+
				" de 1950 e que tem por objetivo determinar se uma máquina é capaz de se portar de forma inteligente.")
	st.markdown("O nome original do Teste de Turing é Jogo da Imitação (Imitation Game no original em inglês)"+
				" e consiste em nada mais que um simples teste de imitação de comportamento."+
				" A ideia é testar se a máquina é capaz de imitar ou replicar o comportamento exibido por um ser humano,"+
				" que é considerado um comportamento inteligente.")
	st.markdown("Antes de iniciar a aplicação do teste, uma pessoa e uma máquina entram em salas distintas, que são trancadas em sequência."+
				" Imagine que você é o aplicador do Teste de Turing e, quando você chega na frente destas salas, você não sabe em qual delas está a máquina."+
				" Então você passa, por debaixo das portas, duas folhas contendo problemas idênticos para que o ser humano e a máquina resolvam"+
				" (no caso específico, este problema são as imagens a serem classificadas entre os dígitos de 0 a 9)."+
				" Cada qual dos competidores resolve o problema e retorna a folha com a resolução."+
				" Se, considerados todos os fatores que você pode observar nesse experimento, for impossível determinar atrás de qual porta está a máquina,"+
				" diz-se que esta máquina passou no Teste de Turing, ou seja, foi capaz de simular um comportamento inteligente.")
	st.image("./tt.png")

	inicio=st.sidebar.button("Fazer o teste")
	modelo=st.sidebar.button("Que modelo é esse?")
	if inicio or modelo:
		if inicio:
			st.session_state.estado="inicio"
		elif modelo:
			st.session_state.estado="modelo"
		st.experimental_rerun()

if st.session_state.estado=='modelo':
	st.subheader("Que modelo é esse?")
	st.markdown("O modelo utilizado para produzir as classificações das imagens, que é utilizado como competidor no Teste de Turing,"+
				" é uma representação do funcionamento da mente de uma criança entre 7 e 12 anos.")
	st.markdown("Os psicólogos e pesquisadores Piaget e Inhelder dividiram o desenvolvimento do pensamento formal da criança em 3 estágios,"+
				" pré-operatório, operatório-concreto e hipotético-dedutivo. No primeiro, a criança é incapaz de diferenciar as fontes dos estímulos que recebe,"+
				" portanto falha em qualquer tarefa de classificação. Já no segundo estágio, a criança é capaz de realizar classificações"+
				" e mapear entradas visuais em modelos mentais. No terceiro, por fim, surge a capacidade de realizar inferências e criar teorias.")
	st.markdown("Na pesquisa realizada, foram modelados e implementados os dois primeiros estágios mencionados,"+
				" utilizando como base um modelo representativo do funcionamento fisiológico do cérebro desenvolvido pelo neurocientista Karl Friston."+
				" O modelo de Friston apresenta finalidades diferentes das desejadas no estudo, por isso ele foi levemente alterado para abarcar o proposto.")
	col1,col2=st.columns(2)
	col1.image("./DCM.png")
	col1.markdown("Modelo proposto por Friston")
	col2.image("./modelo.png")
	col2.markdown("Modelo utilizado no projeto")
	st.markdown("Para cada um dos estágios implementados, esta rede proposta foi altera e reduzida de forma conveniente e"+
				" transformada em redes neurais, de forma que pudessem ser treinadas para oferecer o resultado esperado. Para treinar esses modelos,"+
				" foi utilizado um dataset com imagens de dígitos de 0 a 9 escritos à mão que deveriam ser classficados quanto a qual dígito representam."+
				" Com suficiente tempo de treino, teste e melhoria dos modelos, chegou-se a um estado de representação dos dados e do processo"+
				" que originou o Teste de Turing aqui mostrado e mostrou a eficiência do método. Abaixo estão postas as matrizes de confusão dos modelos obtidos.")
	col1,col2=st.columns(2)
	col1.image("./mc_I.png")
	col1.markdown("Resultado para o primeiro estágio")
	col2.image("./mc_II.png")
	col2.markdown("Resultado para o segundo estágio")
	st.markdown("Como pode ser visto, a rede do primeiro estágio não consegue fazer nada a não ser chutar todos os números como 1, ou seja,"+
				" não é capaz de identificar nenhum dígito, como era esperado de acordo com a teoria de Piaget e Inhelder."+
				" Já a rede do segundo estágio mostra resultados muito fortes, sendo muito acertiva em classficar as imagens do dataset."+
				" Naturalmente, a rede utilizada para competir no Teste de Turing foi a do segundo estágio.")

	tt=st.sidebar.button("O que é o teste de Turing?")
	inicio=st.sidebar.button("Fazer o teste")
	if tt or inicio:
		if tt:
			st.session_state.estado="teste"
		elif inicio:
			st.session_state.estado="inicio"
		st.experimental_rerun()

if st.session_state.estado=='identifica':
	st.markdown("Preencha os campos abaixo com as informações do primeiro jogador")
	nome=st.text_input("Qual seu nome?")
	idade=st.number_input("Qual sua idade?", 0, 100)
	comecar=st.button("Começar o jogo")
	if comecar:
		st.session_state.jogador=(nome, idade)
		n,s=np.random.random_integers(0,9,2)
		i=np.random.randint(10,15)
		st.session_state.maquina=(NOMES[n]+" "+SOBRENOMES[s], str(i))
		st.session_state.estado="imagens"
		st.experimental_rerun()

if st.session_state.estado=='imagens':
	st.sidebar.markdown("Jogando agora: ")
	st.sidebar.markdown(st.session_state.jogador[0]+" (você)")
	st.sidebar.markdown(st.session_state.maquina[0]+" - "+st.session_state.maquina[1]+" anos")
	st.subheader("Primeira fase do Teste")
	imagens=st.session_state.imagens
	st.markdown("Você verá algumas imagens abaixo.")
	cols=st.columns(4)
	escolhas=[]
	for i in range(4):
		for j in range(3):
			fig, ax=plt.subplots()
			ax.axis("off")
			ax.imshow(ndimage.rotate(imagens[3*i+j][0], -90), cmap="Greys")
			fig.gca().invert_xaxis()
			cols[i].pyplot(fig)
			escolhas.append(cols[i].number_input("Qual número é esse?", 0, 9, key=3*i+j))
	acabou=st.button("Terminei!!!")
	if acabou:
		st.session_state.estado='espera'
		resultados={"imagens":imagens, 
					"jog_1":[st.session_state[name] for name in range(12)],
					"jog_2":classifica_imagens(imagens)}
		st.session_state.resultados=resultados
		st.session_state.reais=False
		st.experimental_rerun()

if st.session_state.estado=='espera':
	st.sidebar.markdown("Jogando agora: ")
	st.sidebar.markdown(st.session_state.jogador[0]+" (você)")
	st.sidebar.markdown(st.session_state.maquina[0]+" - "+st.session_state.maquina[1]+" anos")
	st.subheader("Trocando de fases")
	st.markdown("Por favor, deixe que o outro jogador avalie os resultados.")
	proximo=st.button("Já estou aqui")
	if proximo:
		st.session_state.estado='avalia'
		st.experimental_rerun()

if st.session_state.estado=='avalia':
	st.subheader("Segunda fase do Teste")
	st.markdown("Observe as imagens e as classificações feitas para elas.")
	if st.session_state.reais:
		st.markdown(f"Na primeira linha estão as respostas do/da {st.session_state.jogador[0]}, na segunda as da máquina e na terceira estão as respostas esperadas como estão no dataset")
	else:
		st.markdown(f"Nas duas primeiras linhas abaixo da imagem estão as respostas dos competidores, {st.session_state.jogador[0]} e {st.session_state.maquina[0]}. As respostas de cada competidor estão ou na linha de cima ou na de baixo.")
	infos=st.session_state.resultados
	cols=st.columns(4)
	for i in range(4):
		for j in range(3):
			fig, ax=plt.subplots()
			ax.axis("off")
			ax.imshow(ndimage.rotate(infos["imagens"][3*i+j][0], -90), cmap="Greys")
			fig.gca().invert_xaxis()
			cols[i].pyplot(fig)
			cols[i].markdown(f"<div style='text-align: center'> {infos['jog_1'][i*3+j]} </div>", unsafe_allow_html=True)
			cols[i].markdown(f"<div style='text-align: center'> {infos['jog_2'][i*3+j]} </div>", unsafe_allow_html=True)
			if st.session_state.reais:
				cols[i].markdown(f"<div style='text-align: center'> {len(infos['imagens'][i*3+j][1])} </div>", unsafe_allow_html=True)
	if not st.session_state.reais:
		st.markdown("Algum dos conjuntos de respostas foi dado por uma máquina?")
		col1, col2, col3=st.columns(3)
		cima=col1.button("Sim, a máquina deu as respostas de cima")
		baixo=col2.button("Sim, a máquina deu as respostas de baixo")
		nao=col3.button("Não, ambos competidores são humanos")
		if cima or baixo or nao:
			if cima:
				st.session_state.estado='errado'
			if baixo:
				st.session_state.estado='correto'
			if nao:
				st.session_state.estado='nao'
			st.experimental_rerun()

if st.session_state.estado=='errado':
	st.subheader("Resultado do Teste")
	st.markdown("Infelizmente você errou, as respostas da máquina estavam na linha de baixo.")
	st.markdown("A máquina passou no teste de Turing...")
	st.markdown("Será o fim da humanidade?")

if st.session_state.estado=='correto':
	st.subheader("Resultado do Teste")
	st.markdown("Muito bem, você acertou quais eram as respostas da máquina.")
	st.markdown("A máquina não passou no teste de Turing.")
	st.markdown("Parece que a IA ainda não está tão boa.")

if st.session_state.estado=='nao':
	st.subheader("Resultado do Teste")
	st.markdown("Difícil diferenciar, né?")
	st.markdown(f"Na verdade {st.session_state.maquina[0]} era uma máquina e ela passou no teste de Turing...")
	st.markdown("Será o fim da humanidade?")

if st.session_state.estado in ["errado", "correto", "nao"]:
	if st.button("Ver resultados esperados das imagens"):
		st.session_state.estado="avalia"
		st.session_state.reais=True
		st.experimental_rerun()

if st.session_state.estado not in ["inicio", "teste", "modelo"]:
	if st.button("Voltar ao começo"):
		st.session_state.estado = 'inicio'
		st.experimental_rerun()
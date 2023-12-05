<script lang="ts">
	import * as use from '@tensorflow-models/universal-sentence-encoder';
	import * as tf from '@tensorflow/tfjs';

	import type { Tensor2D } from '@tensorflow/tfjs-core';
	import Icon from '@iconify/svelte';

	let name = '';
	type Card = {
		name: string;
		sentences: string;
	};

	let cards: Card[] = [];
	let embeddingMap: Tensor2D[] = [];
	let embeddingTimes: number[] = [];

	let outputs: string = '';
	let outputsIntents: string[] = [];
	function addCard() {
		if (name != '') {
			// check if the list doesnt contain the name
			if (!cards.some((card) => card.name === name)) {
				// add the name to the list
				cards = [...cards, { name, sentences: '' }];
				name = '';
				cards = cards.slice();
			} else {
				alert('Card already exists');
			}
			// reset the name
			name = '';
		}
	}

	function removeCard(name: string) {
		// filter out the card with the name
		cards = cards.filter((card) => card.name !== name);
	}

	function handleKeyPress(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			addCard();
		}
	}

	function cosineSimilarity(a: any, b: any): number {
		const dotProduct = tf.sum(tf.mul(a, b));
		const magnitudeA = tf.sqrt(tf.sum(tf.mul(a, a)));
		const magnitudeB = tf.sqrt(tf.sum(tf.mul(b, b)));
		return dotProduct.div(magnitudeA.mul(magnitudeB)).dataSync()[0];
	}

	function vectorizeInputs() {
		console.log('Vectorizing');
		console.log(cards);
		use
			.load()
			.then(async (model) => {
				// Embed an array of sentences.
				for (let i = 0; i < cards.length; i++) {
					const card = cards[i];
					const sentences = card.sentences.split('\n');
					const start = performance.now();
					const embeddings = await model.embed(sentences);
					embeddingMap[i] = embeddings;
					embeddingTimes[i] = performance.now() - start;
				}
			})
			.finally(() => {
				console.log(embeddingMap);
			});
	}

	async function vectorizeTests() {
		console.log('output');
		console.log(outputs);
		const sentences = outputs.split('\n');
		let model = await use.load();
		for (let i = 0; i < sentences.length; i++) {
			const sentence = sentences[i];
			const embedding = await model.embed([sentence]);
			let maxSimilarity = 0;
			let maxIndex = 0;
			for (let j = 0; j < embeddingMap.length; j++) {
				const cardEmbedding = embeddingMap[j];
				const similarity = cosineSimilarity(embedding, cardEmbedding);
				if (similarity > maxSimilarity) {
					maxSimilarity = similarity;
					maxIndex = j;
				}
			}
			outputsIntents[i] = cards[maxIndex].name;
			console.log(sentence, cards[maxIndex].name);
		}
	}
</script>

<div class="max-w-md mx-auto my-8">
	<div class="flex items-center space-x-2">
		{#if embeddingMap.length == 0}
			<input
				id="nameInput"
				type="text"
				class="border border-gray-300 p-2 w-full"
				bind:value={name}
				on:input={() => {}}
				on:keypress={handleKeyPress}
				placeholder="Enter name"
			/>
		{/if}
		{#if embeddingMap.length > 0}
			<textarea
				class="border border-gray-300 p-2 w-full"
				rows="4"
				placeholder="Enter sentences"
				bind:value={outputs}
			/>
			<button class="btn btn-primary" on:click={() => vectorizeTests()}> Vectorize </button>
			<button
				class="btn btn-primary"
				on:click={() => {
					outputs = '';
					outputsIntents = [];
					embeddingMap = [];
					cards = [];
				}}
			>
				Clear
			</button>
		{/if}
	</div>

	{#if cards.length > 0 && embeddingMap.length == 0}
		<div class="card-list mt-4">
			{#each cards as card, index (card.name)}
				<div class="card">
					<div class="card-title">
						<p>
							{card.name}
						</p>
						<button class="remove-icon" on:click={() => removeCard(card.name)}>
							<Icon icon="mdi:delete" />
						</button>
					</div>
					<textarea
						class="textarea"
						rows="4"
						placeholder="Card content..."
						bind:value={card.sentences}
					></textarea>
				</div>
			{/each}
		</div>
	{/if}

	{#if cards.length >= 2 && embeddingMap.length == 0}
		<div class="flex justify-center mt-4">
			<button class="btn btn-primary" on:click={() => vectorizeInputs()}> Vectorize </button>
		</div>
	{/if}

	{#if outputsIntents.length > 0}
		<table>
			<thead>
				<tr>
					<th>Input</th>
					<th>Output</th>
					<th>Time</th>
				</tr>
			</thead>
			<tbody>
				{#each outputsIntents as output, index}
					<tr>
						<td>{outputs.split('\n')[index]}</td>
						<td>{output}</td>
						<td>{embeddingTimes[index].toFixed(4)}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>

<style lang="postcss">
	:global(html) {
		background-color: theme(colors.blue.400);
	}
	.card-list {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 1rem; /* Adjust the gap between cards as needed */
		width: 100%;
	}

	.card {
		display: flex;
		flex-direction: column;
		border: 2px solid #fff; /* White border */
		padding: 1rem;
		max-width: 800px;
		margin: 0 auto;
	}

	.card-title {
		justify-content: space-evenly;
		display: flex;
		flex-direction: row;
		text-align: center;
		font-size: 1.2rem;
		font-weight: bold;
		margin-bottom: 0.5rem;
	}

	.remove-icon {
		color: #e53e3e; /* Tailwind red-500 */
		cursor: pointer;
		font-size: 1.5rem;
	}

	.textarea {
		width: 100%;
		margin-top: 1rem;
	}
</style>

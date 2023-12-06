<script lang="ts">
	// Imports for Tensorflow related stuff
	import * as use from '@tensorflow-models/universal-sentence-encoder';
	import * as tf from '@tensorflow/tfjs';
	import type { Tensor2D } from '@tensorflow/tfjs-core';

	// Imports from Svelte Material UI
	import Button from '@smui/button';
	import DataTable, { Head, Body, Row, Cell } from '@smui/data-table';

	// Card component
	import Card from '$lib/components/card.svelte';

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
				alert('Intent already exists with this name');
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
		use.load().then(async (model) => {
			// Embed an array of sentences.
			let total = 0;
			const start = performance.now();
			for (let i = 0; i < cards.length; i++) {
				const card = cards[i];
				const sentences = card.sentences.split('\n');
				// remove all empty sentences
				for (let i = 0; i < sentences.length; i++) {
					if (sentences[i] == '') {
						sentences.splice(i, 1);
					}
				}
				total += sentences.length;
				const embeddings = await model.embed(sentences);
				embeddingMap[i] = embeddings;
				embeddingTimes[i] = performance.now() - start;
			}
			console.log('Embedded %s sentences in %s ms', total, performance.now() - start);
		});
	}

	async function vectorizeTests() {
		const sentences = outputs.split('\n');
		// remove all empty sentences
		for (let i = 0; i < sentences.length; i++) {
			if (sentences[i] == '') {
				sentences.splice(i, 1);
			}
		}
		let model = await use.load();
		for (let i = 0; i < sentences.length; i++) {
			const sentence = sentences[i];
			const start = performance.now();
			const embedding = await model.embed([sentence]);
			embeddingTimes[i] = performance.now() - start;
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
			outputsIntents = outputsIntents.slice();
			embeddingTimes = embeddingTimes.slice();
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
				placeholder="Enter Intent Name"
			/>
			<Button disabled={cards.length < 2} variant="raised" on:click={() => vectorizeInputs()}
				>Vectorize</Button
			>
		{/if}

		{#if embeddingMap.length > 0}
			<div class="flex flex-col">
				<div class="flex flex-row p-2 m-4">
					<Button variant="raised" disabled={outputs.length == 0} on:click={() => vectorizeTests()}
						>Vectorize</Button
					>
					<Button
						on:click={() => {
							outputs = '';
							outputsIntents = [];
							embeddingMap = [];
							cards = [];
						}}
						variant="raised"
					>
						Clear
					</Button>
				</div>
				<div>
					<textarea
						class="border border-gray-300 p-2 w-full"
						rows="4"
						placeholder="Enter sentences to tag"
						bind:value={outputs}
					/>
				</div>
			</div>
		{/if}
	</div>

	{#if cards.length > 0 && embeddingMap.length == 0 && outputsIntents.length == 0}
		<div class="flex flex-col w-full">
			{#each cards as card, index (card.name)}
				<Card
					bind:c_sentences={cards[index].sentences}
					bind:c_name={cards[index].name}
					{removeCard}
				/>
			{/each}
		</div>
	{/if}

	{#if outputsIntents.length > 0}
		<DataTable table$aria-label="Outputs list">
			<Head>
				<Row>
					<Cell>Input</Cell>
					<Cell>Output</Cell>
					<Cell>Time (ms)</Cell>
				</Row>
			</Head>
			<Body>
				{#each outputsIntents as output, index}
					<Row>
						<Cell>{outputs.split('\n')[index]}</Cell>
						<Cell>{output}</Cell>
						<Cell>{embeddingTimes[index].toFixed(2)}</Cell>
					</Row>
				{/each}
			</Body>
		</DataTable>
	{/if}
</div>

<style lang="postcss">
</style>

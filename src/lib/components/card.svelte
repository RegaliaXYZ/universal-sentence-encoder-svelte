<script lang="ts">
	import Icon from '@iconify/svelte';
	export let c_name = '';
	export let c_sentences = '';
	export let removeCard: any;
</script>

<div class="flex flex-row border-white border-2 justify-between m-3 p-2">
	<div class="flex flex-col">
		<p>{c_name}</p>
		<button class="remove-icon" on:click={removeCard(c_name)}>
			<Icon icon="mdi:delete" />
		</button>
	</div>
	<br />
	<textarea rows="4" placeholder="Enter sentences" bind:value={c_sentences} />
</div>

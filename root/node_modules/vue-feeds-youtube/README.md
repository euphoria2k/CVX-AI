# vue-feeds-youtube
[![Build Status](https://travis-ci.com/abakermi/vue-feeds-youtube.svg?branch=master)](https://travis-ci.com/abakermi/vue-feeds-youtube) [![npm version](https://badge.fury.io/js/vue-feeds-youtube.svg)](https://badge.fury.io/js/vue-feeds-youtube)


vue component to get youtube feeds

## Installation

### Install via CDN
```html
<script src="https://unpkg.com/vue"></script>
<script src="https://unpkg.com/vue-feeds-youtube@0.0.2"></script>

<script>
  Vue.use(VFeedYoutube.default)
</script>
```

### Install via NPM
```sh
$ npm install vue-feeds-youtube --save
```

#### Register as Component
```js
import Vue from 'vue'
import VFeedYoutube from 'vue-feeds-youtube'
export default {
  name: 'App',
  components: {
    VFeedYoutube
  }
}
```

#### Register as Plugin
```js
import Vue from 'vue'
import VFeedYoutube from 'vue-feeds-youtube'
Vue.use(VFeedYoutube)
```

## Usage

```js
<template>
  <v-feed-youtube channel-id="UC3SdeibuuvF-ganJesKyDVQ">
    
  </v-feed-youtube>
</template>
<script>
import VFeedYoutube from 'vue-feeds-youtube'
export default {
  name: 'App',
  components: {
    VFeedYoutube
  }
}
</script>
```

## Props
|Props|Description|Type|Required|
|-----|-----------|----|--------|
|ChannelId|Youtube channel's id |String|true

## Related

- [rss-parser](https://github.com/rbren/rss-parser) - A small library for turning RSS XML feeds into JavaScript objects.

## License

vue-feeds-youtube is open-sourced software licensed under the [MIT license](http://opensource.org/licenses/MIT)i)
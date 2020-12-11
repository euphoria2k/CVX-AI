import Vue from 'vue'
import router from './router'
import BootstrapVue from "bootstrap-vue"

import App from './App'

import Default from './Layout/Wrappers/baseLayout.vue';
import Pages from './Layout/Wrappers/pagesLayout.vue';
import Multiselect from 'vue-multiselect'
import VueResource from 'vue-resource'

Vue.config.productionTip = false;

Vue.use(BootstrapVue);
Vue.use(VueResource);

Vue.component('multiselect', Multiselect)
Vue.component('default-layout', Default);
Vue.component('userpages-layout', Pages);

new Vue({
  el: '#app',
  router,
  template: '<App/>',
  components: { App }
});
